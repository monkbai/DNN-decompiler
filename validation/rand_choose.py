#!/usr/bin/python3
import subprocess
import os
import sys
import random


class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def cmd(commandline):
    with cd(project_dir):
        print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        print(output)
        return status, output


project_dir = rootdir = "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/"

image_net_dir = "/export/d1/yyuanaq/dataset/IMAGE-NET/val/"


def choose():
    jpg_count = 0
    dirs = os.listdir(image_net_dir)
    random.shuffle(dirs)
    print(dirs)
    for d in dirs:
        d = os.path.join(image_net_dir, d)
        if os.path.isdir(d):
            files = os.listdir(d)
            random.shuffle(files)
            for f in files:
                f = os.path.join(d, f)
                if f.endswith('.JPEG'):
                    status, output = cmd("cp {} ./".format(f))
                    if status == 0:
                        jpg_count += 1
                        break 
            if jpg_count == 100:
                return


if __name__ == '__main__':
    choose()
