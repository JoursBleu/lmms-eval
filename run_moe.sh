#!/bin/bash

set -e
set -v

cd /lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval

export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval:/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers/src

export NPROC_PER_NODE=1
# export NPROC_PER_NODE=4
# export CUDA_VISIBLE_DEVICES=4,5,6,7
unset BLOCK_SIZE
unset SINK_SIZE
unset USE_POS
# export BLOCK_SIZE=4096
# export SINK_SIZE=14
# export USE_POS=1

TASK=longvideobench_val_v-256
MODEL_PATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct

IFS='-' read -ra segments <<< "$TASK"
unset segments[${#segments[@]}-1]
TASK=$(IFS=-; echo "${segments[*]}")

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="runs/eval/$MODEL_NAME/lmms-$TASK"
echo $OUTPUT_DIR

# export LMMS_EVAL_PLUGINS=llava.eval.lmms
# export HF_HOME=$HOME/.cache/huggingface
export HF_HOME=/lpai/volumes/lpai-yharnam-vol-ga/lt/data
export CACHE_DIR=$OUTPUT_DIR/cache_moe
# export CACHE_DIR=$OUTPUT_DIR/cache_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}


torchrun --master_port=25678 --nproc_per_node=$NPROC_PER_NODE \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=${MODEL_PATH},max_pixels=${VIDEO_MAX_PIXELS},use_flash_attention_2=True \
	--tasks $TASK \
	--log_samples \
	--output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true


