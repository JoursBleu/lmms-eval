#!/bin/bash

set -e
set -v

cd /lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval

export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval:/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers/src

# export NPROC_PER_NODE=8
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=1,2
export USE_PAVLM=0
export USE_FRAME=1
export BLOCK_SIZE=16
export SINK_SIZE=4
export USE_POS=1

# export MAX_NUM_FRAMES=32
# export VIDEO_MAX_PIXELS=151200

# export VIDEO_MAX_PIXELS=602112 # (768 * 28 * 28)
# export VIDEO_MAX_PIXELS=1003520 # (1280 * 28 * 28)
# export VIDEO_MAX_PIXELS=1693440 # (2160 * 28 * 28)
# export VIDEO_MAX_PIXELS=3386880 # (4320 * 28 * 28)
export VIDEO_MAX_PIXELS=12845056 # (128000 * 28 * 28 * 0.9)


TASK=longvideobench_val_v
MODEL_PATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen3-VL-30B-A3B-Instruct

MAX_TILES=12

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="runs/eval-time/$MODEL_NAME/lmms-${TASK}-${VIDEO_MAX_PIXELS}"
echo $OUTPUT_DIR

# export LMMS_EVAL_PLUGINS=llava.eval.lmms
# export HF_HOME=$HOME/.cache/huggingface
export HF_HOME=/lpai/volumes/lpai-yharnam-vol-ga/lt/data
export CACHE_DIR=$OUTPUT_DIR/cache_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}


torchrun --master_port=25678 --nproc_per_node=$NPROC_PER_NODE \
	-m lmms_eval \
	--model qwen3_vl \
	--model_args pretrained=${MODEL_PATH},max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
	--tasks $TASK \
	--log_samples \
	--output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true


