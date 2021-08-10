#!/usr/bin/python3
import os
import time
import json
import numpy as np
import subprocess


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
        # print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output


project_dir = './'

def validate_glow(prog1: str, prog2: str, input_dir: str):
    prog1 = os.path.abspath(prog1)
    prog2 = os.path.abspath(prog2)
    input_dir = os.path.abspath(input_dir)

    files = os.listdir(input_dir)
    for f in files:
        if f.endswith('.bin'):
            f = os.path.join(input_dir, f)
            status1, output1 = cmd("{} {}".format(prog1, f))
            status2, output2 = cmd("{} {}".format(prog2, f))
            if output1 != output2:
                print(f)
                print(output1)
                print(output2)
                print('inconsistency')
                break


def validate_tvm(prog1: str, prog2: str, input_dir: str):
    prog1 = os.path.abspath(prog1)
    prog2 = os.path.abspath(prog2)
    input_dir = os.path.abspath(input_dir)

    files = os.listdir(input_dir)
    fail_count = 0
    for f in files:
        if f.endswith('.bin'):
            f = os.path.join(input_dir, f)
            status1, output1 = cmd("{} {}".format(prog1, f))
            status2, output2 = cmd("{} {}".format(prog2, f))
            output1 = output1[:output1.find('timing')].strip()  # skip timing
            output2 = output2[:output2.find('timing')].strip()  # skip timing
            if output1 != output2:
                print(prog1)
                print(prog2)
                print(f)
                print(output1)
                print(output2)
                print('inconsistency')
                fail_count += 1
    print('fail count', fail_count)


def embedding_validate_glow(prog1: str, prog2: str, input_dir: str):
    prog1 = os.path.abspath(prog1)
    prog2 = os.path.abspath(prog2)
    input_dir = os.path.abspath(input_dir)

    files = os.listdir(input_dir)
    for f in files:
        if f.endswith('.txt'):
            f = os.path.join(input_dir, f)
            with open(f, 'r') as f:
                j_txt = f.read()
                list_obj = json.loads(s=j_txt)
                arr_obj = np.array(list_obj, dtype=np.int32)
            status1, output1 = cmd("{} {} {} {} {} {} {} {}".format(prog1, arr_obj[0][0], arr_obj[1][0], arr_obj[2][0], arr_obj[3][0], arr_obj[4][0], arr_obj[5][0], arr_obj[6][0]))
            status2, output2 = cmd("{} {} {} {} {} {} {} {}".format(prog2, arr_obj[0][0], arr_obj[1][0], arr_obj[2][0], arr_obj[3][0], arr_obj[4][0], arr_obj[5][0], arr_obj[6][0]))
            if output1 != output2:
                print(f)
                print(output1)
                print(output2)
                print('inconsistency')
                break


def embedding_validate_tvm(prog1: str, prog2: str, input_dir: str):
    prog1 = os.path.abspath(prog1)
    prog2 = os.path.abspath(prog2)
    input_dir = os.path.abspath(input_dir)

    files = os.listdir(input_dir)
    fail_count = 0
    for f in files:
        if f.endswith('.txt'):
            f = os.path.join(input_dir, f)
            
            with open(f, 'r') as f:
                j_txt = f.read()
                list_obj = json.loads(s=j_txt)
                arr_obj = np.array(list_obj, dtype=np.int32)
            status1, output1 = cmd("{} {} {} {} {} {} {} {}".format(prog1, arr_obj[0][0], arr_obj[1][0], arr_obj[2][0], arr_obj[3][0], arr_obj[4][0], arr_obj[5][0], arr_obj[6][0]))
            status2, output2 = cmd("{} {} {} {} {} {} {} {}".format(prog2, arr_obj[0][0], arr_obj[1][0], arr_obj[2][0], arr_obj[3][0], arr_obj[4][0], arr_obj[5][0], arr_obj[6][0]))
            output1 = output1[:output1.find('timing')].strip()  # skip timing
            output2 = output2[:output2.find('timing')].strip()  # skip timing
            if output1 != output2:
                print(prog1)
                print(prog2)
                print(f)
                print(output1)
                print(output2)
                print('inconsistency')
                fail_count += 1
    print('fail count', fail_count)

if __name__ == '__main__':
    # =======================================================
    # Vgg and ResNet
    #validate_glow("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_glow/vgg16_strip.out", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/vgg16_glow/vgg16_glow_rebuild/vgg16_glow_rebuild.out", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")
    #validate_glow("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_glow/resnet18_strip.out", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/resnet18_glow/resnet18_glow_rebuild/resnet18_glow_rebuild.out", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")

    #validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/vgg16_tvm_O0_strip", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/vgg16_tvm_O0/vgg16_tvmO0_rebuild/build/demo_static", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")
    #validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O3/vgg16_tvm_O3_strip", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/vgg16_tvm_O3/vgg16_tvmO3_rebuild/build/demo_static", 
    #              "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")
    validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O0/resnet18_tvm_O0_strip", 
                  "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/resnet18_tvm_O0/resnet18_tvmO0_rebuild/build/demo_static", 
                  "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")
    # validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/resnet18_tvm_O3/resnet18_tvm_O3_strip", 
    #               "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/resnet18_tvm_O3/resnet18_tvmO3_rebuild/build/demo_static", 
    #               "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/")

    # =======================================================
    # Embedding modle
    # embedding_validate_glow("/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_glow/embedding_build/embedding.out", 
    #                         "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_glow/embedding_rebuild/embedding_rebuild.out", 
    #                         "/export/d1/zliudc/DLE_Decompiler/TVM/embedding_input/")
    
    # embedding_validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_tvm_O0/embedding_build/build/demo_static", 
    #                        "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_tvm_O0/embedding_rebuild/build/demo_static", 
    #                        "/export/d1/zliudc/DLE_Decompiler/TVM/embedding_input/")
    # embedding_validate_tvm("/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_tvm_O3/embedding_build/build/demo_static", 
    #                        "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/embedding_tvm_O3/embedding_rebuild/build/demo_static", 
    #                        "/export/d1/zliudc/DLE_Decompiler/TVM/embedding_input/")

    # =======================================================
    # nnfusion

