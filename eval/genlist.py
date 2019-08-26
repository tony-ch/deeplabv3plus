#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os,sys
import re
import argparse

"""
生成列表
"""

def getlist(curpath, fout):
    dirlist=os.listdir(curpath)
    dirlist = sorted(dirlist)
    for dir in dirlist:
        if os.path.isfile(curpath+"/"+dir):
            fout.write(curpath+"/"+dir+"\n")
        if os.path.isdir(curpath+"/"+dir):
            getlist(curpath+"/"+dir, fout)
    #for f in dirlist:
        #if re.match(r'.*g$',f):
            #name = os.path.splitext(f)
            #print curpath+f
            #fout.write(curpath+"/"+f+"\n")

def parse_args_fun(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_dir', default=os.getcwd()+"/img",           
        help='img dir to handle (default os.getcwd()+"/img)')
    parser.add_argument('--list_file', default='list/list.txt', help='file path to save list(default list/list.txt)')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args_fun(sys.argv)
    fout = open(args.list_file,'w')
    getlist(args.img_dir,fout)
