import tensorflow as tf


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b, text_a_label=None,
               text_b_label=None, class_label=None):
    """Constructs a InputExample.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.text_a_label = text_a_label
    self.text_b_label = text_b_label
    self.class_label = class_label


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length.

  Copied from bert/run_classifier.py
  """

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def truncate_length_and_warn(tokens_a, tokens_b, max_seq_length, guid):
  # Modifies `tokens_a` and `tokens_b` in place so that the total
  # length is less than the specified length.
  # Account for [CLS], [SEP], [SEP] with "- 3"
  if len(tokens_a) + len(tokens_b) > max_seq_length - 3:
    tf.logging.info("Truncate Example %s. Total len=%d." % (guid, len(tokens_a) + len(tokens_b)))
    input_text_too_long = True
  else:
    input_text_too_long = False

  _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

  return input_text_too_long


def get_token_label_ids(token_labels_a, token_labels_b, max_seq_length):
  token_label_ids = []
  token_label_ids.append(0)

  for token_label in token_labels_a:
    token_label_ids.append(token_label)

  token_label_ids.append(0)

  for token_label in token_labels_b:
    token_label_ids.append(token_label)

  token_label_ids.append(0)

  while len(token_label_ids) < max_seq_length:
    token_label_ids.append(0)

  assert len(token_label_ids) == max_seq_length
  return token_label_ids


def get_start_end_pos(class_type, token_label_ids, max_seq_length):
  if class_type == 'copy_value' and 1 not in token_label_ids:
    raise ValueError('Copy value but token_label not detected.')
  if class_type != 'copy_value':
    start_pos = 0
    end_pos = 0
  else:
    start_pos = token_label_ids.index(1)
    end_pos = max_seq_length - 1 - token_label_ids[::-1].index(1)
    # tf.logging.info('token_label_ids: %s' % str(token_label_ids))
    # tf.logging.info('start_pos: %d' % start_pos)
    # tf.logging.info('end_pos: %d' % end_pos)
    for i in range(max_seq_length):
      if i >= start_pos and i <= end_pos:
        assert token_label_ids[i] == 1
      else:
        assert token_label_ids[i] == 0
  return start_pos, end_pos


def get_bert_input(tokens_a, tokens_b, max_seq_length, tokenizer):
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []

  tokens.append("[CLS]")
  segment_ids.append(0)

  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)

  tokens.append("[SEP]")
  segment_ids.append(0)

  for token in tokens_b:
    tokens.append(token)
    segment_ids.append(1)

  tokens.append("[SEP]")
  segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  return tokens, input_ids, input_mask, segment_ids


def fully_connect(logits, weights, bias=None, activation=None):
  out_logits = tf.matmul(logits, weights)
  if bias is not None:
    out_logits = tf.nn.bias_add(out_logits, bias)
  if activation == 'relu':
    out_logits = tf.nn.relu(out_logits)
  return out_logits


def fully_connect_layers(input_layer, list_weights, list_bias):
  """Fully conntect multiple layers, with
  (1) input layer unchanged.
  (2) all layers have relu activation except for the last layer.
  """
  if len(list_weights) == 1:
    logits = fully_connect(input_layer, list_weights[0], list_bias[0])
  else:
    logits = fully_connect(input_layer, list_weights[0], list_bias[0], activation='relu')
    if len(list_weights) > 2:
      for l_idx in range(1, len(list_weights) - 1):
        logits = fully_connect(logits, list_weights[l_idx], list_bias[l_idx], activation='relu')
    logits = fully_connect(logits, list_weights[-1], list_bias[-1])
  return logits

