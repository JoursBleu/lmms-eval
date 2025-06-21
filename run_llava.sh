#!/bin/bash

set -e
set -v

cd /lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval

# export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval:/lpai/volumes/lpai-yharnam-vol-ga/lt/LLaVA-NeXT:/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers-vila/src
export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval:/lpai/volumes/lpai-yharnam-vol-ga/lt/LLaVA-NeXT

pip3 uninstall transformers -y
pip3 install /lpai/volumes/lpai-yharnam-vol-ga/lt/transformers-vila

# export NPROC_PER_NODE=8
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=5
# export NPROC_PER_NODE=1
# export CUDA_VISIBLE_DEVICES=1
export USE_PAVLM=1
# export USE_FRAME=1
export BLOCK_SIZE=4096
export SINK_SIZE=0
export USE_POS=1
# unset BLOCK_SIZE
# unset SINK_SIZE
# unset USE_POS

export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'


TASK=longvideobench_val_v
MODEL_PATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/LLaVA-Video-7B-Qwen2

IFS='-' read -ra segments <<< "$TASK"
unset segments[${#segments[@]}-1]

MODEL_NAME=$(basename $MODEL_PATH)
OUTPUT_DIR="runs/eval/$MODEL_NAME/test-$TASK"
echo $OUTPUT_DIR

rm -rf ${OUTPUT_DIR}

# export LMMS_EVAL_PLUGINS=llava.eval.lmms
# export HF_HOME=$HOME/.cache/huggingface
export HF_HOME=/lpai/volumes/lpai-yharnam-vol-ga/lt/data
# export CACHE_DIR=$OUTPUT_DIR/cache_frame_sysframe_${USE_POS}
export CACHE_DIR=$OUTPUT_DIR/cache_${USE_PAVLM}_${BLOCK_SIZE}_${SINK_SIZE}_${USE_POS}


torchrun --master_port=25678 --nproc_per_node=$NPROC_PER_NODE \
    -m lmms_eval \
	--model llava_onevision \
	--model_args pretrained=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=128 \
	--tasks $TASK \
	--batch_size 1 \
	--log_samples \
	--output_path $OUTPUT_DIR


mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true


