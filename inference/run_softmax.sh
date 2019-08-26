#!/usr/bin/env bash
#-*- coding:utf-8 -*-

MODEL_DIR="models/deeplabv3_shen_on_mars_batch_cleanv3/" # mpdel dir
MODEL='frozen_inference_graph_batch_sz769_20000_softmax.pb'
RESULTS="result/matting_test/shen_iter20000_sz769" # dir to save seg results
IMG_DIR="/home/tony/app/models/research/deeplab/datasets/matting/data/matting_human_half/" # img dir
CURRENT_DIR=$(pwd)


LISTS="${CURRENT_DIR}/list/list.txt"

# generate list file in img
echo "generating list file in:"
echo "  ${IMG_DIR}"
python genlist.py \
    --img_dir="${IMG_DIR}" \
    --list_file="${LISTS}" #\--prefix_path
echo "done"


echo ""
echo "running inference"
python inference_softmax.py \
    --model_dir="${MODEL_DIR}" \
    --model_file="${MODEL}" \
    --size=769 \
    --save_dir="${RESULTS}" \
    --list_file="${LISTS}" \
    --img_root="${IMG_DIR}" \
    --mode='softmax_multi'
