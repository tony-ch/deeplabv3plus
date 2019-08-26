#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

cd ..
# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#   python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download PASCAL VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
# cd "${WORK_DIR}/${DATASET_DIR}"
# sh download_and_convert_voc2012.sh
# python build_mars_data.py

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
DATASET_FOLDER="shen_matting"
EXP_FOLDER="exp_shen_matting_on_mars_sz769_cleanv2/train_on_train_set_20000"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"

TFRECORD_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord_cleanv2"
INIT_CHECKPOINT="${INIT_FOLDER}/mars/model.ckpt"
CROP_SIZE=769
COSTUM_DATASET="shen_matting"
BATCH_SIZE=1
NUM_ITERATIONS=20000
RES_MODE="softmax_multi"
NUM_CLASSES=3

mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

cd "${CURRENT_DIR}"

#CUDA_VISIBLE_DEVICES="0,1,2"
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="${CROP_SIZE},${CROP_SIZE}" \
  --train_batch_size=${BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=False \
  --dataset="${COSTUM_DATASET}" \
  --initialize_last_layer=False \
  --tf_initial_checkpoint="${INIT_CHECKPOINT}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TFRECORD_DIR}"

# Run evaluation. This performs eval over the full val split (1449 images) and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.
# python "${WORK_DIR}"/eval.py \
#   --logtostderr \
#   --eval_split="val" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --eval_crop_size="${CROP_SIZE},${CROP_SIZE}" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --eval_logdir="${EVAL_LOGDIR}" \
#   --dataset_dir="${TFRECORD_DIR}" \
#   --max_number_of_evaluations=1

# # Visualize the results.
# python "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="test" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --vis_crop_size="${CROP_SIZE},${CROP_SIZE}" \
#   --dataset="${COSTUM_DATASET}" \
#   --colormap_type="${COSTUM_DATASET}" \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${TFRECORD_DIR}" \
#   --max_number_of_iterations=1  \
#   --also_save_raw_predictions=true

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph_batch_sz${CROP_SIZE}_${NUM_ITERATIONS}.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=${NUM_CLASSES} \
  --crop_size=${CROP_SIZE} \
  --crop_size=${CROP_SIZE} \
  --inference_scales=1.0 \
  --res_mode="${RES_MODE}" 

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
