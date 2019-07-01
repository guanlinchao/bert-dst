#!/usr/bin/env bash

# TASK can be "dstc2_clean", "woz2", "sim-m", or "sim-r"
TASK="sim-m"
# Directory for dstc2-clean, woz_2.0, sim-M, or sim-R, which contains json files
DATA_DIR=/path/to/sim-M
# Directory of the pre-trained [BERT-Base, Uncased] model
PRETRAINED_BERT=/path/to/uncased_L-12_H-768_A-12
# Output directory of trained checkpoints, evaluation and prediction outputs
OUTPUT_DIR=/path/to/output
# DSET can be "dev" or "test"
DSET="dev"

# Comma separated list of checkpoint steps to be evaluated.
for num in {0..12000..1000}; do
  CKPT_NUM="$CKPT_NUM,$num"
done

python main.py \
  --task_name=${TASK} \
  --do_eval=true \
  --do_predict=true \
  --max_seq_length=180 \
  --eval_set=$DSET \
  --eval_ckpt=$CKPT_NUM \
  --data_dir=$DATA_DIR \
  --vocab_file=${PRETRAINED_BERT}/vocab.txt \
  --bert_config_file=${PRETRAINED_BERT}/bert_config.json \
  --init_checkpoint=${PRETRAINED_BERT}/bert_model.ckpt \
  --output_dir=$OUTPUT_DIR \
  2>&1 | tee -a $OUTPUT_DIR/eval.log


python metric_bert_dst.py \
${TASK} \
"$OUTPUT_DIR/pred_res.${DSET}*json"
