#!/usr/bin/python3
import subprocess
import os

# export PATH=/home/hwangdz/export-d2/binutils/build-2.36/gcc-7_-O3_-g_/bin:$PATH

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
        # print(output)
        return status, output


def run(prog_path):
    with cd(project_dir):
        # print(prog_path)
        proc = subprocess.Popen(prog_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()  # stderr: summary, stdout:  each statement
        return stdout, stderr


project_dir = rootdir = "/export/d1/zliudc/DLE_Decompiler/TVM/scripts/op_comprehensive/inception_glow_2021/"
cmd_str = "time python3 ./inception_glow_decompile.py"
rm_cmd_str = "rm /export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/inception_v1/0102_slice.log"

count = 0

def once():
    global project_dir, count 
    file_size = os.path.getsize("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/Glow-2021/inception_v1/0102_slice.log")
    if file_size < 1800000:
        status, output = cmd(rm_cmd_str)
        status, output = cmd(cmd_str)
        count += 1
        print('re-generate times:', count)
        return False
    else:
        return True


def find_one():
    while True:
        if once():
            break
        else:
            continue


if __name__ == '__main__':
    find_one()
