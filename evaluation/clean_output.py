import os
import re
import sys
import subprocess

home_dir = '/home/DNN-decompiler/evaluation'


def clean_output():
    for subdir, dirs, files in os.walk(home_dir):
        for file in files:
            mat1 = re.match(r"\d{4,4}\.log", file)  # 0010.log
            mat2 = re.match(r"\d{4,4}\.\w+_\d+\.json", file)  # 0010.weights_0.json
            mat3 = re.match(r"\w*meta_data\.json", file)  # 
            mat4 = re.match(r"\w*topo_list\.json", file)
            if mat1 or mat2 or mat3 or mat4:
                #print os.path.join(subdir, file)
                filepath = os.path.join(subdir, file)
                print("rm {}".format(filepath))
                status, output = subprocess.getstatusoutput("rm {}".format(filepath))
                if status:
                    print(output)
            else:
                pass

if __name__ == '__main__':
    clean_output()
