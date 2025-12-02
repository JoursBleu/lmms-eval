#!/bin/bash

set -e
set -v

cd /lpai/volumes/lpai-yharnam-lx-my/lt/lmms-eval

export PYTHONPATH=/lpai/volumes/lpai-yharnam-lx-my/lt/lmms-eval:/lpai/volumes/lpai-yharnam-lx-my/lt/transformers-qwen3/src

# export NPROC_PER_NODE=8
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=4
export USE_PAVLM=1
export USE_FRAME=1
export BLOCK_SIZE=32
export SINK_SIZE=32
export USE_POS=1

export FPS_MAX_FRAMES=768
# export VIDEO_MAX_PIXELS=151200
# export VIDEO_MAX_PIXELS=602112 # (768 * 28 * 28)
# export VIDEO_MAX_PIXELS=1003520 # (1280 * 28 * 28)
# export VIDEO_MAX_PIXELS=1693440 # (2160 * 28 * 28)
# export VIDEO_MAX_PIXELS=3386880 # (4320 * 28 * 28)
# export VIDEO_MAX_PIXELS=90316800 # (128000 * 28 * 28 * 0.9)
export VIDEO_MAX_PIXELS=180633600 # (256000 * 28 * 28 * 0.9)

export API_TYPE=openai
export OPENAI_API_URL=https://lpai-inference-miyun.inner.chj.cloud/inference/lpai-lmp/lmms/v1/chat/completions
export OPENAI_API_KEY="abc"

export DECORD_EOF_RETRY_MAX=50000
# TASK=mvbench
# TASK=egoschema_subset
# TASK=videomme
# TASK=detailed_test
# TASK=mme_videoocr
# TASK=longvideobench_val_v
TASK=tempcompass

# MODEL_PATH=/lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-VL-2B-Instruct
# MODEL_PATH=/lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-VL-4B-Instruct
# MODEL_PATH=/lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-VL-8B-Instruct
MODEL_PATH=/lpai/volumes/lpai-yharnam-lx-my/lt/models/Qwen3-VL-32B-Instruct

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="runs/eval-time/$MODEL_NAME/lmms-${TASK}-${FPS_MAX_FRAMES}-${VIDEO_MAX_PIXELS}"
echo $OUTPUT_DIR

# export LMMS_EVAL_PLUGINS=llava.eval.lmms
# export HF_HOME=$HOME/.cache/huggingface
export HF_HOME=/lpai/volumes/lpai-yharnam-lx-my/lt/data
export CACHE_DIR=$OUTPUT_DIR/cache_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}


python -m lmms_eval \
--model qwen3_vl \
--model_args pretrained=${MODEL_PATH},max_pixels=${VIDEO_MAX_PIXELS},attn_implementation=flash_attention_2,interleave_visuals=False,max_num_frames=${FPS_MAX_FRAMES} \
--tasks $TASK \
--log_samples \
--force_simple \
--output_path $OUTPUT_DIR

# torchrun --master_port=25678 --nproc_per_node=$NPROC_PER_NODE \
	# -m lmms_eval \
	# --model qwen3_vl \
	# --model_args pretrained=${MODEL_PATH},max_pixels=${VIDEO_MAX_PIXELS},attn_implementation=flash_attention_2,interleave_visuals=False,max_num_frames=${FPS_MAX_FRAMES} \
	# --tasks $TASK \
	# --log_samples \
	# --force_simple \
	# --output_path $OUTPUT_DIR

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}.jsonl || true


