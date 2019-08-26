#!/usr/bin/env bash
#-*- coding:utf-8 -*-

MODELS="models/deeplabv3_teeth/" # mpdel dir
RESULTS="./result/deeplab_teeth/" # dir to save seg results
IMG_DIR="./data/teeth_test/" # img dir
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
python inference.py \
    --model_dir="${MODELS}" \
    --save_dir="${RESULTS}" \
    --list_file="${LISTS}" \
    --img_root="${IMG_DIR}" \
    --show_vis=True