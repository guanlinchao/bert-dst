#!/usr/bin/env bash

# TASK can be "dstc2_clean", "woz2", "sim-m", or "sim-r"
TASK="dstc2_clean"
# Directory for dstc2-clean, woz_2.0, sim-M, or sim-R, which contains json files
DATA_DIR=/path/to/dstc2-clean
# Directory of the pre-trained [BERT-Base, Uncased] model
PRETRAINED_BERT=/path/to/uncased_L-12_H-768_A-12
# Output directory of trained checkpoints
OUTPUT_DIR=/path/to/output

mkdir -p $OUTPUT_DIR
python main.py \
  --task_name=${TASK} \
  --do_train=true \
  --train_batch_size=16 \
  --slot_value_dropout=0.0 \
  --max_seq_length=180 \
  --data_dir=$DATA_DIR \
  --vocab_file=${PRETRAINED_BERT}/vocab.txt \
  --bert_config_file=${PRETRAINED_BERT}/bert_config.json \
  --init_checkpoint=${PRETRAINED_BERT}/bert_model.ckpt \
  --learning_rate=2e-5 \
  --num_train_epochs=100 \
  --output_dir=$OUTPUT_DIR \
  2>&1 | tee -a $OUTPUT_DIR/train.log
