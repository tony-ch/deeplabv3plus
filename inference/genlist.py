#!/usr/bin/env python
#-*- coding:utf-8 -*-
import os,sys
import re
import argparse

"""
生成列表
"""

# def getlist(curpath, fout):
#     dirlist=os.listdir(curpath)
#     dirlist = sorted(dirlist)
#     for dir in dirlist:
#         if os.path.isfile(curpath+"/"+dir):
#             fout.write(curpath+"/"+dir+"\n")
#         if os.path.isdir(curpath+"/"+dir):
#             getlist(curpath+"/"+dir, fout)
    #for f in dirlist:
        #if re.match(r'.*g$',f):
            #name = os.path.splitext(f)
            #print curpath+f
            #fout.write(curpath+"/"+f+"\n")
def getlist(relative_path, fout, root):
    curpath = root+relative_path
    dirlist=os.listdir(curpath)
    dirlist = sorted(dirlist)
    for dir in dirlist:
        if os.path.isfile(curpath+dir):
            fout.write(relative_path+dir+"\n")
        if os.path.isdir(curpath+dir):
            getlist(relative_path+dir+"/", fout, root)

def parse_args_fun(argv):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_dir', default="img/",           
        help='img dir to handle (default "img/)')
    parser.add_argument('--list_file', default='list/list.txt', help='file path to save list(default list/list.txt)')
    parser.add_argument('-p','--prefix_path', action="store_true", help='gen list and add "img_dir" as prefix')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args_fun(sys.argv)
    fout = open(args.list_file,'w')
    if not os.path.isdir(args.img_dir):
        sys.stderr.write("not valid img_path")
        exit(1)
    abs_img_dir=os.path.abspath(args.img_dir)+"/"
    if args.prefix_path:
        getlist(abs_img_dir,fout,"")
    else:
        getlist("",fout,abs_img_dir)
