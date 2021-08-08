#!/usr/bin/python3
import subprocess
import os
import sys



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


# project_dir = rootdir = r"C:\Users\44916\Desktop\TVM_binaries\TVM_binaries\O3"
project_dir = rootdir = r"C:\Users\44916\Desktop\TVM_binaries\GLOW_binaries"
ida_exe = r"C:\WorkSpace\tools\ollydbg_IDAPRO\IDA_Pro_v7.0_Portable\ida64.exe"
ida_lst_cmd = ida_exe + r' -A -S"C:\Users\44916\Desktop\TVM_binaries\ida_lst.py" {}'
ida_str_cmd = ida_exe + r' -A -S"C:\Users\44916\Desktop\TVM_binaries\ida_string.py" {}'
# ida_tmp_dir = r"C:\Users\44916\Desktop"


def main_lst(work_dir: str):
    work_dir = os.path.abspath(work_dir)
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith('.out') and 'GLOW_binaries' in root:
                pass
            elif '.' not in file:
                pass
            else:
                continue
            # 
            lst_path = file_path + '.lst'
            tmp_path = os.path.join(root, 'tmp.lst')
            if not os.path.exists(lst_path):
                status, output = cmd(ida_lst_cmd.format(file_path))
                print(output)
                # input('before copy')
                status, output = subprocess.getstatusoutput(r'copy ' + tmp_path+ ' '+ lst_path)
                status, output = subprocess.getstatusoutput(r'del ' + tmp_path)
                # input('continue?')


def main_string(work_dir: str):
    work_dir = os.path.abspath(work_dir)
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            if file.endswith('.lst') or file.endswith('.txt') or file.endswith('.i64') or file.endswith('.py'):
                continue
            
            str_path = file_path + '.txt'
            tmp_path = os.path.join(root, 'strings.txt')
            if not os.path.exists(str_path):
                status, output = cmd(ida_str_cmd.format(file_path))
                print(output)
                # input('before copy')
                status, output = subprocess.getstatusoutput(r'copy ' + tmp_path+ ' '+ str_path)
                status, output = subprocess.getstatusoutput(r'del ' + tmp_path)
                # input('continue?')


if __name__ == '__main__':
    main_string(rootdir)
