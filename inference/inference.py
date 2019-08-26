#!/usr/bin/env python
#-*- coding:utf-8 -*-

#@title Imports
from __future__ import print_function
import os
import time

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


#@title Helper methods

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', 'models/deeplabv3_portrait_on_mars_10000_softmax/', 'Directory of frozen inference graph.')

flags.DEFINE_string('save_dir', 'portrait_result/deeplabv3_portrait_on_mars/', 'Directory of segmentation to be saved.')

flags.DEFINE_string('img_root', "./portrait/image/test/", 'Path of img.')

flags.DEFINE_string('list_file', 'list/list.txt', 'Path of image list.')

flags.DEFINE_boolean('show_vis', True,
                     'Run segmetation as show vis result.')

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'

  def __init__(self, model_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    graph_path = os.path.join(model_path, self.FROZEN_GRAPH_NAME)
    with tf.gfile.FastGFile(graph_path,'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      #resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    #seg_map = np.resize(seg_map,(height,width))
    return image, seg_map

def mask_colormap():
  return np.array([
      [0,0,0],
      [255,255,255]], dtype=int)

def human_parse_colormap():
  return np.array([
    [0,0,0],
    [128,0,0],
    [254,0,0],
    [0,85,0],
    [169,0,51],
    [254,85,0],
    [0,0,85],
    [0,119,220],
    [85,85,0],
    [0,85,85],
    [85,51,0],
    [52,86,128],
    [0,128,0],
    [0,0,254],
    [51,169,220],
    [0,254,254],
    [85,254,169],
    [169,254,85],
    [254,254,0],
    [254,169,0]
  ] ,dtype=int)

def create_label_colormap():
  """Creates a label colormap used in segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  
  colormap[0:2,:]= mask_colormap()
  # colormap[0:20,:] = human_parse_colormap()
  return colormap

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

# LABEL_NAMES = np.asarray([
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
# ])
LABEL_NAMES = np.asarray([
  'person', 
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def vis_segmentation(image, seg_map, seg_image):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 4))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 6])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    width, height = image.size

    plt.subplot(grid_spec[1])
    #seg_image = label_to_color_image(seg_map).astype(np.uint8)
    #seg_image = Image.fromarray(seg_map.astype(np.int8),mode='P')
    #seg_image = seg_image.resize((width,height))
    plt.imshow(seg_map)
    plt.axis('off')
    plt.title('raw segmentation map')


    plt.subplot(grid_spec[2])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[3])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    plt.grid('off')
    plt.show()

def run_segmentation(MODEL, path, root):
  """Inferences DeepLab model and visualizes result."""
  try:
    orignal_im = Image.open(root+path)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + path)
    return

  print('running deeplab on image %s...' % path)

  resized_im, seg_map = MODEL.run(orignal_im)
  width,height = orignal_im.size
  size = (width,height)
  pil_image = Image.fromarray(seg_map.astype(dtype=np.uint8))
  pil_image = pil_image.resize(size)

  file_name = os.path.basename(path)[:-4]

  if not os.path.isabs(path):#图片列表给出的项目是相对img_root的路径
    file_name=path[:-4]
    tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_dir+"/raw/"+path))
    tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_dir+"/color/"+path))

  with tf.gfile.Open('%s/%s.jpg' % (FLAGS.save_dir+"/raw", file_name ), mode='w') as f:
    pil_image.save(f, 'JPEG')
  seg_image = label_to_color_image(seg_map).astype(dtype=np.uint8)
  seg_image = Image.fromarray(seg_image)
  seg_image = seg_image.resize((width,height))
  with tf.gfile.Open('%s/%s.jpg' % (FLAGS.save_dir+"/color", file_name ), mode='w') as f:
    seg_image.save(f, 'JPEG')
  
  if FLAGS.show_vis:
    vis_segmentation(resized_im, seg_map, seg_image)

def main(unused_argv):
  #@title Sload models {display-mode: "form"}
  sstime=time.time()
  print('loading DeepLab model...')
  MODEL = DeepLabModel(FLAGS.model_dir)
  print('model loaded successfully!')


  #@title Run on sample images {display-mode: "form"}
  tf.gfile.MakeDirs(FLAGS.save_dir)
  tf.gfile.MakeDirs(FLAGS.save_dir+"/raw")
  tf.gfile.MakeDirs(FLAGS.save_dir+"/color")
  stime = time.time()
  
  root = os.path.abspath(FLAGS.img_root)+"/"
  with open(FLAGS.list_file,'r') as f_list:
    for image_path in f_list:
      image_path = image_path.strip()
      run_segmentation(MODEL, image_path, root)
      
  etime = time.time()
  print("begin:", sstime)
  print("begin inf:", stime)
  print("end:", etime)
  print("load model:", stime - sstime, "s")
  print("infer:", etime - stime, "s")


if __name__ == '__main__':
  #flags.mark_flag_as_required('checkpoint_dir')
  #flags.mark_flag_as_required('vis_logdir')
  #flags.mark_flag_as_required('dataset_dir')
  tf.app.run()