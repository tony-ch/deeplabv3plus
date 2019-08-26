#!/usr/bin/env python
#-*- encoding:utf-8 -*-

import numpy as np
import time
import argparse
import sys
from PIL import Image


eps = 2.2204e-16


def pixelAccuracy(imPred,imAnno):
    
    # Remove classes from unlabeled pixels in gt image. 
    pixel_labeled = np.sum(imAnno>=0)
    pixel_correct = np.sum( (imAnno==imPred)*(imAnno>=0) )
    pixel_accuracy = pixel_correct*1.0/pixel_labeled
    return pixel_correct,pixel_labeled,pixel_accuracy

def IOU(imPred,imAnno,numclass):
    # imPred = imPred.flatten()
    # imAnno = imAnno.flatten()
    
    # Remove classes from unlabeled pixels in label image. 
    #imPred = imPred*(imAnno>0)

    # bin edge [0,1) [1,2) ... [numclass,numclass+1]
    # bins = np.arange(numclass+2)

    #  Compute area intersection
    intersection = imPred*(imPred==imAnno)
    area_intersection = np.histogram(intersection,bins=numclass+1,range=(0,numclass))[0]
    x=np.sum(imPred==2)
    # Compute area union
    area_pred = np.histogram(imPred, bins=numclass+1,range=(0,numclass))[0]
    area_anno = np.histogram(imAnno, bins=numclass+1,range=(0,numclass))[0]
    area_union = area_pred + area_anno - area_intersection

    # Remove unlabeled bin
    area_intersection = area_intersection[1:]
    area_union = area_union[1:]

    return area_intersection,area_union




def readlist(list_file):
    f = open(list_file)
    lists = f.readlines()
    lists = [ x.strip() for x in lists]
    return lists



def eval_all(anno_list,pred_list,numclass,data_class):
    img_num = len(anno_list)
    area_intersection = np.zeros([numclass,img_num])
    area_union = np.zeros([numclass,img_num])
    pixel_accuracy = np.zeros(img_num)
    pixel_correct = np.zeros(img_num)
    pixel_labeled = np.zeros(img_num)

    for i in range(img_num):
        im_anno = np.array(Image.open(anno_list[i]))
        im_pred = np.array(Image.open(pred_list[i]))
        # im_anno = im_anno + 1
        # im_pred = im_pred + 1
        # im_anno[im_anno==255] = 0
        im_pred = im_pred//255
        # im_anno = im_anno//255

        if im_anno.ndim!=2:
            sys.stderr.write('Label image [%s] should be a gray-scale image!\n' % (pred_list[i]))
            continue
        if im_anno.shape!=im_pred.shape:
            sys.stderr.write('Lable image [%s] should have the same size as label image! Resizing...' % (pred_list[i]))
            im_pred = np.resize(im_pred,im_pred.shape)
        
        area_intersection[:,i],area_union[:,i] = IOU(im_pred,im_anno,numclass)
        pixel_correct[i],pixel_labeled[i],pixel_accuracy[i] = pixelAccuracy(im_pred,im_anno)
        sys.stdout.write('Evaluating %d/%d: Pixel-wise accuracy: %2.2f%%\n' % (i+1,img_num,pixel_accuracy[i]*100.0))
    
    IoU = np.sum(area_intersection,axis=1)*1.0/np.sum(area_union+eps,axis=1)
    mean_IoU = np.mean(IoU)
    accuracy = np.sum(pixel_correct)/np.sum(pixel_labeled)

    sys.stdout.write('==== Summary IoU ====\n')
    for i in range(numclass):
        sys.stdout.write('%3d %16s: %.4f\n'%(i,data_class[i],IoU[i]))
    sys.stdout.write('Mean IoU over %d classes: %.4f\n' % (numclass, mean_IoU))
    sys.stdout.write('Pixel-wise Accuracy: %2.2f%%\n' % (accuracy*100))

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_list', default='./list/anno_list.txt',           
        help='img anno list')
    parser.add_argument('--pred_list', default='./list/pred_list.txt',           
        help='img seg list')
    parser.add_argument('--numclass', default='1',           
        help='numclass list')
    parser.add_argument('--data_class', default='./list/data_class.txt',           
        help='data class list')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArgs(sys.argv)
    
    anno_list = readlist(args.anno_list)
    pred_list = readlist(args.pred_list)
    data_class = readlist(args.data_class)
    numclass = int(args.numclass)
    eval_all(anno_list,pred_list,numclass,data_class)

