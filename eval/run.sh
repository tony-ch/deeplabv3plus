#!/usr/bin/env sh

#echo "gen anno list"
#python genlist.py \
#    --img_dir="/home/tony/data/coco/val/mask/" \
#    --list_file="./list/anno_list.txt" 

#echo "gen pred list"
#python genlist.py \
#    --img_dir="/home/tony/app/models/research/deeplab/inference/result/matting_test/iter60000_sz513/" \
#    --list_file="./list/pred_list.txt" 

python eval.py \
    --anno_list="./list/anno_list.txt" \
    --pred_list="./list/pred_list.txt" \
    --numclass=1 \
    --data_class="./list/data_class.txt" \
    2>&1 | tee eval.log
