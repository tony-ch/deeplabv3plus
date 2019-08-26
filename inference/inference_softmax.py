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
import os

CUDA_VISIBLE_DEVICES = [0]
CUDA_MEM_LIMIT = 8000.0 # MB

#@title Helper methods

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', 'models/deeplabv3_shen_on_mars_40000_sz769_softmax_batch/', 'Directory of frozen inference graph.')

flags.DEFINE_string('model_file', 'frozen_inference_graph_batch_sz769.pb', 'Name of frozen inference graph.')

flags.DEFINE_integer('size', 769, 'Size of input image.')

flags.DEFINE_string('save_dir', 'result/test/', 'Directory of segmentation to be saved.')

flags.DEFINE_string('img_root', "/home/tony/app/models/research/deeplab/datasets/matting/data/matting_human_half/", 'Path of img.')

flags.DEFINE_string('list_file', '/home/tony/app/models/research/deeplab/datasets/matting/data/matting_human_half/test.txt', 'Path of image list.')

flags.DEFINE_float('lower_thres', 0.001, "lower threshold for gen trimap")

flags.DEFINE_float('upper_thres', 0.999, "lower threshold for gen trimap")

flags.DEFINE_integer('batch_size', 8, "batch size")

flags.DEFINE_string('mode', 'softmax_multi','using softmax or argmax')

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = FLAGS.size
    PAD = 127.5
    FROZEN_GRAPH_NAME = FLAGS.model_file

    def parse(self, line,qargs):
        '''解析一行nvidia-smi返回的csv格式文本'''
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
        power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
        process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}
    
    def query_gpu(self, qargs=[]):
        '''查询GPU信息'''
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [self.parse(line,qargs) for line in results]

    def cal_mem_fraction(self):
        '''计算每个GPU占用的比例'''
        gpu_info = self.query_gpu()
        totalMEM = 0
        limitMEM = CUDA_MEM_LIMIT - 300 # 申请的大小总会比限制的值多300M,将限制调低
        for gid in CUDA_VISIBLE_DEVICES:
            totalMEM+=gpu_info[gid]['memory.total']
        return limitMEM/totalMEM

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        os.environ['CUDA_VISIBLE_DEVICES']=str(CUDA_VISIBLE_DEVICES)[1:-1]
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

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=self.cal_mem_fraction())
        config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(graph=self.graph,config=config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            #resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: image})
        return batch_seg_map
from PIL import ExifTags

def handle_orientation(image):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    exif = image._getexif()
    if not exif:
        return image
    exif=dict(exif.items())
    if(len(exif)==0 or not exif.has_key(orientation)):
        return image
    if exif[orientation] == 3:
        image=image.rotate(180, expand=True)
    elif exif[orientation] == 6:
        image=image.rotate(270, expand=True)
    elif exif[orientation] == 8:
        image=image.rotate(90, expand=True)
    return image

def make_trimap(seg_map):
    low = FLAGS.lower_thres
    up = FLAGS.upper_thres
    seg_map[seg_map<=low] = 0
    seg_map[seg_map>up] = 1
    seg_map[(seg_map<=up)*(seg_map>low)] = 0.5
    return seg_map * 255

def save_trimap(trimap_image, path, size):
    file_name = os.path.basename(path)[:-4]
    if not os.path.isabs(path):#图片列表给出的项目是相对img_root的路径
        file_name=path[:-4]
        tf.gfile.MakeDirs(os.path.dirname(FLAGS.save_dir+"/"+path))

    with tf.gfile.Open('%s/%s.jpg' % (FLAGS.save_dir+"/", file_name ), mode='w') as f:
        trimap_image.save(f, 'JPEG')
def pad_im(im,model_input_size,pad):
    im = im - pad
    h,w,c = im.shape
    im = np.pad(im,((0,max(model_input_size-h,0)),(0,max(model_input_size-w,0)),(0,0)),'constant')
    im = im + pad
    return im,(h,w)

def read_im(path,model_input_size):
    try:
        orignal_im = Image.open(path)
        orignal_im = handle_orientation(orignal_im)
        # plt.imshow(orignal_im)
        # plt.show()
        width, height = orignal_im.size
        resize_ratio = 1.0 * model_input_size / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = orignal_im.convert('RGB').resize(target_size, Image.ANTIALIAS)
        return np.asarray(resized_image), (width,height)
    except IOError:
        print('Cannot retrieve image. Please check url: ' + path)
        return None

def main(unused_argv):
    #@title Sload models {display-mode: "form"}
    sstime=time.time()
    print('loading DeepLab model...')
    MODEL = DeepLabModel(FLAGS.model_dir)
    print('model loaded successfully!')


    #@title Run on sample images {display-mode: "form"}
    tf.gfile.MakeDirs(FLAGS.save_dir)
    stime = time.time()

    root = os.path.abspath(FLAGS.img_root)+"/"
    with open(FLAGS.list_file,'r') as f_file:
        f_list = f_file.readlines()
        f_cnt = len(f_list)
        for i in range(0,f_cnt,FLAGS.batch_size):
            batch = []
            for j in range(FLAGS.batch_size):
                if i+j>=f_cnt:
                    break
                resized_image, ori_size = read_im(root+f_list[i+j].strip().split()[0],MODEL.INPUT_SIZE)
                padded_im,crop_size = pad_im(resized_image,MODEL.INPUT_SIZE,MODEL.PAD)
                batch.append({'path':f_list[i+j].strip().split()[0], 'im':padded_im, 'ori_sz':ori_size, 'crop_sz':crop_size})
            batch_image = [b['im'] for b in batch]
            seg_map = MODEL.run(batch_image)
            print("run on batch:")
            for j in range(len(batch)):
                print("\t%s"%batch[j]['path'])
                crop_sz = batch[j]['crop_sz']
                mask = seg_map[j][0:crop_sz[0],0:crop_sz[1]]
                if FLAGS.mode == 'softmax_single': # softmax on all channel and return single channel
                    trimap = make_trimap(mask)
                elif FLAGS.mode == 'argmax':
                    trimap = mask * 0.5 * 255
                    plt.imshow(trimap.astype('uint8'))
                    plt.show()
                elif FLAGS.mode == 'softmax_multi': # softmax and return multi channel
                    trimap = np.argmax(mask,-1)*0.5*255
                    fg = mask[:,:,-1]*255
                    print(mask.shape)
                    plt.imshow(trimap)
                    plt.show()
                    plt.imshow(fg)
                    plt.show()
                    # fg = Image.fromarray(fg.astype('uint8')).resize(ori_size, Image.ANTIALIAS)
                else :
                    raise Exception('not support')
                trimap = Image.fromarray(trimap.astype('uint8')).resize(ori_size, Image.ANTIALIAS)
                save_trimap(trimap,batch[j]['path'], batch[j]['ori_sz'])
            
    etime = time.time()
    print("begin:", sstime)
    print("begin inf:", stime)
    print("end:", etime)
    print("load model:", stime - sstime, "s")
    print("infer:", etime - stime, "s")
    print(len(f_list)/(etime-stime),"fps")


if __name__ == '__main__':
    tf.app.run()
