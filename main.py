"""Model definition for BERT-DST. Modified from bert/run_classifier.py"""
import collections
import json
import numpy as np
import os
import sys

import tensorflow as tf

import dataset_dstc2
import dataset_sim
import util

flags = tf.flags
FLAGS = flags.FLAGS

sys.path.append("/path/to/bert")

import modeling
import optimization
import run_classifier
import tokenization

## BERT-DST params

flags.DEFINE_string(
    "eval_set", "dev",
    "Which set to be evaluated: dev or test.")

flags.DEFINE_string(
    "eval_ckpt", "",
    "comma seperated ckpt numbers to be evaluated.")

flags.DEFINE_integer(
    "num_class_hidden_layer", 0,
    "Number of prediction layers in class prediction.")

flags.DEFINE_integer(
    "num_token_hidden_layer", 0,
    "Number of prediction layers in class prediction.")

flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate for BERT representations.")

flags.DEFINE_bool(
    "location_loss_for_nonpointable", False,
    "Whether the location loss for none or dontcare is contributed towards total loss.")

flags.DEFINE_float("class_loss_ratio", 0.8,
                   "The ratio applied on class loss in total loss calculation."
                   "Should be a value in [0.0, 1.0]"
                   "The ratio applied on token loss is (1-class_loss_ratio).")

flags.DEFINE_float("slot_value_dropout", 0.0,
                   "The rate that targeted slot value was replaced by [UNK].")


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               start_pos,
               end_pos,
               class_label_id,
               is_real_example=True,
               guid="NONE"):
    self.guid = guid
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_pos = start_pos
    self.end_pos = end_pos
    self.class_label_id = class_label_id
    self.is_real_example = is_real_example


class Dstc2Processor(object):
  class_types = ['none', 'dontcare', 'copy_value', 'unpointable']
  slot_list = ['area', 'food', 'price range']

  def get_train_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'dstc2_train_en.json'), self.slot_list, 'train')

  def get_dev_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'dstc2_validate_en.json'), self.slot_list, 'dev',
      use_asr_hyp=1, exclude_unpointable=False)

  def get_test_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'dstc2_test_en.json'), self.slot_list, 'test',
      use_asr_hyp=1, exclude_unpointable=False)


class Woz2Processor(object):
  class_types = ['none', 'dontcare', 'copy_value', 'unpointable']
  slot_list = ['area', 'food', 'price range']

  def get_train_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'woz_train_en.json'), self.slot_list, 'train')

  def get_dev_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'woz_validate_en.json'), self.slot_list, 'dev',
      use_asr_hyp=0, exclude_unpointable=False)

  def get_test_examples(self, data_dir):
    return dataset_dstc2.create_examples(
      os.path.join(data_dir, 'woz_test_en.json'), self.slot_list, 'test',
      use_asr_hyp=0, exclude_unpointable=False)


class SimMProcessor(object):
  class_types = ['none', 'dontcare', 'copy_value']
  slot_list = ['date', 'movie', 'time', 'num_tickets', 'theatre_name']

  def get_train_examples(self, data_dir):
    return dataset_sim.create_examples(
      os.path.join(data_dir, 'train.json'), self.slot_list, 'train')

  def get_dev_examples(self, data_dir):
    return dataset_sim.create_examples(
      os.path.join(data_dir, 'dev.json'), self.slot_list, 'dev')

  def get_test_examples(self, data_dir):
    return dataset_sim.create_examples(
      os.path.join(data_dir, 'test.json'), self.slot_list, 'test')


class SimRProcessor(SimMProcessor):
  slot_list = ['category', 'rating', 'num_people', 'location',
               'restaurant_name', 'time', 'date', 'price_range', 'meal']


def tokenize_text_and_label(text, text_label_dict, slot, tokenizer):
  joint_text_label = [0 for _ in text_label_dict[slot]] # joint all slots' label
  for slot_text_label in text_label_dict.values():
    for idx, label in enumerate(slot_text_label):
      if label == 1:
        joint_text_label[idx] = 1

  text_label = text_label_dict[slot]
  tokens = []
  token_labels = []
  for token, token_label, joint_label in zip(text, text_label, joint_text_label):
    token = tokenization.convert_to_unicode(token)
    sub_tokens = tokenizer.tokenize(token)
    if FLAGS.slot_value_dropout == 0.0 or joint_label == 0:
      tokens.extend(sub_tokens)
    else:
      rn_list = np.random.random_sample((len(sub_tokens),))
      for rn, sub_token in zip(rn_list, sub_tokens):
        if rn > FLAGS.slot_value_dropout:
          tokens.append(sub_token)
        else:
          tokens.append('[UNK]')

    token_labels.extend([token_label for _ in sub_tokens])
  assert len(tokens) == len(token_labels)
  return tokens, token_labels


def convert_single_example(ex_index, example, slot_list, class_types, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, run_classifier.PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        start_pos={slot: 0 for slot in slot_list},
        end_pos={slot: 0 for slot in slot_list},
        class_label_id={slot: 0 for slot in slot_list},
        is_real_example=False,
        guid="NONE")

  class_label_id_dict = {}
  start_pos_dict = {}
  end_pos_dict = {}
  for slot in slot_list:
    tokens_a, token_labels_a = tokenize_text_and_label(
      example.text_a, example.text_a_label, slot, tokenizer)
    tokens_b, token_labels_b = tokenize_text_and_label(
      example.text_b, example.text_b_label, slot, tokenizer)

    input_text_too_long = util.truncate_length_and_warn(
      tokens_a, tokens_b, max_seq_length, example.guid)

    if input_text_too_long:
      if ex_index < 10:
        if len(token_labels_a) > len(tokens_a):
          tf.logging.info('    tokens_a truncated labels: %s' % str(token_labels_a[len(tokens_a):]))
        if len(token_labels_b) > len(tokens_b):
          tf.logging.info('    tokens_b truncated labels: %s' % str(token_labels_b[len(tokens_b):]))

      token_labels_a = token_labels_a[:len(tokens_a)]
      token_labels_b = token_labels_b[:len(tokens_b)]

    assert len(token_labels_a) == len(tokens_a)
    assert len(token_labels_b) == len(tokens_b)
    token_label_ids = util.get_token_label_ids(
      token_labels_a, token_labels_b, max_seq_length)

    class_label_id_dict[slot] = class_types.index(example.class_label[slot])
    start_pos_dict[slot], end_pos_dict[
      slot] = util.get_start_end_pos(
      example.class_label[slot], token_label_ids,
      max_seq_length)

  tokens, input_ids, input_mask, segment_ids = util.get_bert_input(tokens_a,
                                                                   tokens_b,
                                                                   max_seq_length,
                                                                   tokenizer)

  if ex_index < 10:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("start_pos: %s" % str(start_pos_dict))
    tf.logging.info("end_pos: %s" % str(end_pos_dict))
    tf.logging.info("class_label_id: %s" % str(class_label_id_dict))


  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      start_pos=start_pos_dict,
      end_pos=end_pos_dict,
      class_label_id=class_label_id_dict,
      is_real_example=True,
      guid=example.guid)
  return feature, input_text_too_long


def file_based_convert_examples_to_features(
    examples, slot_list, class_types, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  total_cnt = 0
  too_long_cnt = 0

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature, input_text_too_long = convert_single_example(ex_index, example, slot_list, class_types,
                                     max_seq_length, tokenizer)
    total_cnt += 1
    if input_text_too_long:
      too_long_cnt += 1

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    for slot in slot_list:
      features["start_pos_%s" % slot] = create_int_feature([feature.start_pos[slot]])
      features["end_pos_%s" % slot] = create_int_feature([feature.end_pos[slot]])
      features["class_label_id_%s" % slot] = create_int_feature([feature.class_label_id[slot]])
    features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
    features["guid"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature.guid.encode('utf-8')]))

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  tf.logging.info("========== %d out of %d examples have text too long" % (too_long_cnt, total_cnt))
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder, slot_list):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
      "guid": tf.FixedLenFeature([], tf.string)
  }
  for slot in slot_list:
    name_to_features["start_pos_%s" % slot] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_pos_%s" % slot] = tf.FixedLenFeature([], tf.int64)
    name_to_features["class_label_id_%s" % slot] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, slot_list, features, num_class_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  input_ids = features["input_ids"]
  input_mask = features["input_mask"]
  segment_ids = features["segment_ids"]

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  class_output_layer = model.get_pooled_output()
  token_output_layer = model.get_sequence_output()

  token_output_shape = modeling.get_shape_list(token_output_layer, expected_rank=3)
  batch_size = token_output_shape[0]
  seq_length = token_output_shape[1]
  hidden_size = token_output_shape[2]

  # Define prediction variables
  class_proj_layer_dim = [hidden_size]
  for idx in range(FLAGS.num_class_hidden_layer):
    class_proj_layer_dim.append(64)
  class_proj_layer_dim.append(num_class_labels)

  token_proj_layer_dim = [hidden_size]
  for idx in range(FLAGS.num_token_hidden_layer):
    token_proj_layer_dim.append(64)
  token_proj_layer_dim.append(2)

  if is_training:
    # I.e., 0.1 dropout
    class_output_layer = tf.nn.dropout(class_output_layer,
                                       keep_prob=(1 - FLAGS.dropout_rate))
    token_output_layer = tf.nn.dropout(token_output_layer,
                                       keep_prob=(1 - FLAGS.dropout_rate))
  total_loss = 0
  per_slot_per_example_loss = {}
  per_slot_class_logits = {}
  per_slot_start_logits = {}
  per_slot_end_logits = {}
  for slot in slot_list:
    start_pos = features["start_pos_%s" % slot]
    end_pos = features["end_pos_%s" % slot]
    class_label_id = features["class_label_id_%s" % slot]
    slot_scope_name = "slot_%s" % slot
    if slot == 'price range':
      slot_scope_name = "slot_price"
    with tf.variable_scope(slot_scope_name):
      class_list_output_weights = []
      class_list_output_bias = []

      for l_idx in range(len(class_proj_layer_dim) - 1):
        dim_in = class_proj_layer_dim[l_idx]
        dim_out = class_proj_layer_dim[l_idx + 1]
        class_list_output_weights.append(tf.get_variable(
          "class/output_weights_%d" % l_idx, [dim_in, dim_out],
          initializer=tf.truncated_normal_initializer(stddev=0.02)))
        class_list_output_bias.append(tf.get_variable(
          "class/output_bias_%d" % l_idx, [dim_out],
          initializer=tf.zeros_initializer()))

      token_list_output_weights = []
      token_list_output_bias = []

      for l_idx in range(len(token_proj_layer_dim) - 1):
        dim_in = token_proj_layer_dim[l_idx]
        dim_out = token_proj_layer_dim[l_idx + 1]
        token_list_output_weights.append(tf.get_variable(
          "token/output_weights_%d" % l_idx, [dim_in, dim_out],
          initializer=tf.truncated_normal_initializer(stddev=0.02)))
        token_list_output_bias.append(tf.get_variable(
          "token/output_bias_%d" % l_idx, [dim_out],
          initializer=tf.zeros_initializer()))

      with tf.variable_scope("loss"):
        class_logits = util.fully_connect_layers(class_output_layer,
                                                 class_list_output_weights,
                                                 class_list_output_bias)
        one_hot_class_labels = tf.one_hot(class_label_id,
                                          depth=num_class_labels,
                                          dtype=tf.float32)
        class_loss = tf.losses.softmax_cross_entropy(
          one_hot_class_labels, class_logits, reduction=tf.losses.Reduction.NONE)

        token_is_pointable = tf.cast(tf.equal(class_label_id, 2), dtype=tf.float32)

        token_output_layer = tf.reshape(token_output_layer,
                                        [batch_size * seq_length, hidden_size])
        token_logits = util.fully_connect_layers(token_output_layer,
                                                 token_list_output_weights,
                                                 token_list_output_bias)
        token_logits = tf.reshape(token_logits, [batch_size, seq_length, 2])
        token_logits = tf.transpose(token_logits, [2, 0, 1])
        unstacked_token_logits = tf.unstack(token_logits, axis=0)
        (start_logits, end_logits) = (
        unstacked_token_logits[0], unstacked_token_logits[1])

        def compute_loss(logits, positions):
          one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
          log_probs = tf.nn.log_softmax(logits, axis=1)
          loss = -tf.reduce_sum(one_hot_positions * log_probs, axis=1)
          return loss

        token_loss = (compute_loss(start_logits, start_pos) + compute_loss(end_logits, end_pos)) / 2.0 # per example
        if not FLAGS.location_loss_for_nonpointable:
          token_loss *= token_is_pointable

        per_example_loss = FLAGS.class_loss_ratio * class_loss + (1-FLAGS.class_loss_ratio) * token_loss

        total_loss += tf.reduce_sum(per_example_loss)
        per_slot_per_example_loss[slot] = per_example_loss
        per_slot_class_logits[slot] = class_logits
        per_slot_start_logits[slot] = start_logits
        per_slot_end_logits[slot] = end_logits
  return (total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits)


def model_fn_builder(bert_config, slot_list, num_class_labels, init_checkpoint,
                     learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_slot_per_example_loss, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits) = create_model(
      bert_config, is_training, slot_list, features, num_class_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_slot_per_example_loss, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, is_real_example):
        metric_dict = {}
        per_slot_correctness = {}
        for slot in slot_list:
          per_example_loss = per_slot_per_example_loss[slot]
          class_logits = per_slot_class_logits[slot]
          start_logits = per_slot_start_logits[slot]
          end_logits = per_slot_end_logits[slot]

          class_label_id = features['class_label_id_%s' % slot]
          start_pos = features['start_pos_%s' % slot]
          end_pos = features['end_pos_%s' % slot]

          class_prediction = tf.cast(tf.argmax(class_logits, axis=1), tf.int32)
          class_correctness = tf.cast(tf.equal(class_prediction, class_label_id), dtype=tf.float32)
          class_accuracy = tf.metrics.mean(
            tf.reduce_sum(
              class_correctness * is_real_example) / tf.reduce_sum(
            is_real_example))

          token_is_pointable = tf.cast(tf.equal(class_label_id, 2),
                                       dtype=tf.float32)
          start_prediction = tf.cast(tf.argmax(start_logits, axis=1), tf.int32)
          start_correctness = tf.cast(tf.equal(start_prediction, start_pos),
                                      dtype=tf.float32)
          end_prediction = tf.cast(tf.argmax(end_logits, axis=1), tf.int32)
          end_correctness = tf.cast(tf.equal(end_prediction, end_pos),
                                      dtype=tf.float32)
          token_correctness = start_correctness * end_correctness
          token_accuracy = tf.metrics.mean(
            tf.reduce_sum(
              token_correctness * token_is_pointable) / tf.reduce_sum(
              token_is_pointable))

          total_corretness = class_correctness * (token_is_pointable * token_correctness + (1-token_is_pointable))
          total_accuracy = tf.metrics.mean(
            tf.reduce_sum(total_corretness) * is_real_example / tf.reduce_sum(is_real_example))
          loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
          metric_dict['eval_accuracy_class_%s' % slot] = class_accuracy
          metric_dict['eval_accuracy_token_%s' % slot] = token_accuracy
          metric_dict['eval_accuracy_%s' % slot] = total_accuracy
          metric_dict['eval_loss_%s' % slot] = loss
          per_slot_correctness[slot] = total_corretness
        goal_correctness = tf.reduce_prod(
          tf.stack(
            [correctness for correctness in
             per_slot_correctness.values()],
            axis=1),
          axis=1)
        goal_accuracy = tf.metrics.mean(tf.reduce_sum(goal_correctness * is_real_example) / tf.reduce_sum(
          is_real_example))
        metric_dict['eval_accuracy_goal'] = goal_accuracy
        return metric_dict

      eval_metrics = (metric_fn,
                      [per_slot_per_example_loss, features, per_slot_class_logits, per_slot_start_logits, per_slot_end_logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      predictions_dict = {"guid": features["guid"]}
      for slot in slot_list:
        slot_scope_name = "slot_%s" % slot
        if slot == 'price range':
          slot_scope_name = "slot_price"
        with tf.variable_scope(slot_scope_name):
          class_prediction = tf.argmax(per_slot_class_logits[slot], axis=1)
          start_prediction = tf.argmax(per_slot_start_logits[slot], axis=1)
          end_prediction = tf.argmax(per_slot_end_logits[slot], axis=1)

          predictions_dict["class_prediction_%s" % slot] = class_prediction
          predictions_dict["class_label_id_%s" % slot] = features["class_label_id_%s" % slot]
          predictions_dict["start_prediction_%s" % slot] = start_prediction
          predictions_dict["start_pos_%s" % slot] = features["start_pos_%s" % slot]
          predictions_dict["end_prediction_%s" % slot] = end_prediction
          predictions_dict["end_pos_%s" % slot] = features["end_pos_%s" % slot]
          predictions_dict["input_ids_%s" % slot] = features["input_ids"]

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=predictions_dict,
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
    "dstc2_clean": Dstc2Processor,
    "woz2": Woz2Processor,
    "sim-m": SimMProcessor,
    "sim-r": SimRProcessor,
  }

  tokenization.validate_case_matches_checkpoint(
    do_lower_case=True, init_checkpoint=FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  slot_list = processor.slot_list
  class_types = processor.class_types
  num_class_labels = len(class_types)
  if task_name in ['woz2', 'dstc2_clean']:
    num_class_labels -= 1

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=None,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      slot_list=slot_list,
      num_class_labels=num_class_labels,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, slot_list, class_types, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        slot_list=slot_list)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    if FLAGS.eval_set == 'dev':
      eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    else:
      eval_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(run_classifier.PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.%s.tf_record" % FLAGS.eval_set)
    file_based_convert_examples_to_features(
        eval_examples, slot_list, class_types, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        slot_list=slot_list)
    output_eval_file = os.path.join(FLAGS.output_dir,
                                    "eval_res.%s.json" % FLAGS.eval_set)
    if tf.gfile.Exists(output_eval_file):
      with tf.gfile.GFile(output_eval_file) as f:
        eval_result = json.load(f)
    else:
      eval_result = []

    ckpt_nums = [num.strip() for num in FLAGS.eval_ckpt.split(',') if num.strip() != ""]
    for ckpt_num in ckpt_nums:
      result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps,
                                  checkpoint_path=os.path.join(FLAGS.output_dir,
                                  "model.ckpt-%s" % ckpt_num))
      result_dict = {k: float(v) for k, v in result.items()}
      eval_result.append(result_dict)
      tf.logging.info("***** Eval results for %s set *****", FLAGS.eval_set)
      for key in sorted(result.keys()):
        tf.logging.info("%s = %s", key, str(result[key]))
    if len(eval_result) > 0:
      with tf.gfile.GFile(output_eval_file, "w") as f:
        json.dump(eval_result, f, indent=2)

  if FLAGS.do_predict:
    if FLAGS.eval_set == 'dev':
      predict_examples = processor.get_dev_examples(FLAGS.data_dir)
    else:
      predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(run_classifier.PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "pred.%s.tf_record" % FLAGS.eval_set)
    file_based_convert_examples_to_features(predict_examples, slot_list, class_types,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        slot_list=slot_list)

    ckpt_nums = [num for num in FLAGS.eval_ckpt.split(',') if num != ""]
    for ckpt_num in ckpt_nums:
      result = estimator.predict(input_fn=predict_input_fn,
                                 checkpoint_path=os.path.join(FLAGS.output_dir,
                                    "model.ckpt-%s" % ckpt_num))

      output_predict_file = os.path.join(FLAGS.output_dir,
        "pred_res.%s.%08d.json" % (FLAGS.eval_set, int(ckpt_num)))
      with tf.gfile.GFile(output_predict_file, "w") as f:
        num_written_ex = 0
        tf.logging.info("***** Predict results for %s set *****", FLAGS.eval_set)
        list_prediction = []
        for (i, prediction) in enumerate(result):
          # Str feature is encoded as bytes, which is not JSON serializable.
          # Hence convert to str.
          prediction["guid"] = prediction["guid"].decode("utf-8").split("-")
          for slot in slot_list:
            start_pd = prediction['start_prediction_%s' % slot]
            start_gt = prediction['start_pos_%s' % slot]
            end_pd = prediction['start_prediction_%s' % slot]
            end_gt = prediction['end_pos_%s' % slot]
            # TF uses int64, which is not JSON serializable.
            # Hence convert to int.
            prediction['class_prediction_%s' % slot] = int(prediction['class_prediction_%s' % slot])
            prediction['class_label_id_%s' % slot] = int(prediction['class_label_id_%s' % slot])
            prediction['start_prediction_%s' % slot] = int(start_pd)
            prediction['start_pos_%s' % slot] = int(start_gt)
            prediction['end_prediction_%s' % slot] = int(end_pd)
            prediction['end_pos_%s' % slot] = int(end_gt)
            prediction["input_ids_%s" % slot] = list(map(int, prediction["input_ids_%s" % slot].tolist()))
            input_tokens = tokenizer.convert_ids_to_tokens(prediction["input_ids_%s" % slot])
            prediction["slot_prediction_%s" % slot] = ' '.join(input_tokens[start_pd:end_pd+1])
            prediction["slot_groundtruth_%s" % slot] = ' '.join(input_tokens[start_gt:end_gt + 1])
          list_prediction.append(prediction)
          if i >= num_actual_predict_examples:
            break
          num_written_ex += 1
        json.dump(list_prediction, f, indent=2)
      assert num_written_ex == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
