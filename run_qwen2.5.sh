#!/bin/bash

set -e
set -v

cd /lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval

export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval:/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers/src

# export NPROC_PER_NODE=8
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=6
# unset BLOCK_SIZE
# unset SINK_SIZE
# unset USE_POS
export USE_PAVLM=0
export USE_FRAME=1
export BLOCK_SIZE=16
export SINK_SIZE=128
export USE_POS=1

# export MAX_NUM_FRAMES=32
export VIDEO_MAX_PIXELS=10000

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'


TASK=longvideobench_val_v
MODEL_PATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="runs/eval-test/$MODEL_NAME/lmms-${TASK}-${MAX_NUM_FRAMES}-${VIDEO_MAX_PIXELS}"
echo $OUTPUT_DIR
rm -rf ${OUTPUT_DIR}

# export LMMS_EVAL_PLUGINS=llava.eval.lmms
# export HF_HOME=$HOME/.cache/huggingface
export HF_HOME=/lpai/volumes/lpai-yharnam-vol-ga/lt/data
# export CACHE_DIR=$OUTPUT_DIR/cache_${USE_PAVLM}_16frame_sysframe_${USE_POS}
export CACHE_DIR=$OUTPUT_DIR/cache_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}


torchrun --master_port=25678 --nproc_per_node=$NPROC_PER_NODE \
	-m lmms_eval \
	--model qwen2_5_vl \
	--model_args pretrained=${MODEL_PATH},max_pixels=${VIDEO_MAX_PIXELS},use_flash_attention_2=True,max_num_frames=32 \
	--tasks $TASK \
	--log_samples \
	--output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true


