import os
import re
import sys
import subprocess

btd_data_dir = '/home/BTD-data/'


def clean_trace():
    for subdir, dirs, files in os.walk(btd_data_dir):
        for file in files:
            mat = re.match(r"\d{4,4}(_rev)?(_slice)?\.log", file)
            if mat:
                #print os.path.join(subdir, file)
                filepath = os.path.join(subdir, file)
                print("rm {}".format(filepath))
                status, output = subprocess.getstatusoutput("rm {}".format(filepath))
                if status:
                    print(output)
            else:
                pass

if __name__ == '__main__':
    clean_trace()
