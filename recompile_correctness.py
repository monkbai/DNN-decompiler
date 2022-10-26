import os
import re
import sys
import subprocess
from subprocess import Popen, PIPE, STDOUT

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
        # print(project_dir)
        # print(commandline)
        status, output = subprocess.getstatusoutput(commandline)
        # print(output)
        return status, output

project_dir = os.path.abspath('./')
input_img = os.path.abspath('./cat.bin')
pass_count = 0

def run_rebuilt_model(script_path: str, in_data=input_img):
    global project_dir
    target_dir = os.path.dirname(script_path)
    tmp_dir = project_dir
    project_dir = target_dir

    script_name = os.path.basename(script_path)
    status, output = cmd("python3.6 {} {}".format(script_name, in_data))

    project_dir = tmp_dir

    if status == 0:
        return output
    else:
        print(output)

def run_original_model(exe_path: str, in_data=input_img):
    status, output = cmd("{} {}".format(exe_path, in_data))
    if status == 0:
        return output
    else:
        print(output)


def compare_tvm(rebuilt_output: str, origi_output: str):
    global pass_count
    mat = re.search(r"Result: (\d+)", rebuilt_output)
    if mat:
        rebuilt_label = int(mat.group(1))

    mat = re.search(r"The maximum position in output vector is: (\d+),", origi_output)
    if mat:
        origi_label = int(mat.group(1))

    if rebuilt_label == origi_label:
        pass_count += 1
        return True
    else:
        return False


def compare_glow(rebuilt_output: str, origi_output: str):
    global pass_count
    mat = re.search(r"Result: (\d+)", rebuilt_output)
    if mat:
        rebuilt_label = int(mat.group(1))

    mat = re.search(r"Result: (\d+)", origi_output)
    if mat:
        origi_label = int(mat.group(1))

    if rebuilt_label == origi_label:
        pass_count += 1
        return True
    else:
        return False


def compare_fasttext(rebuilt_output: str, origi_output: str):
    global pass_count
    mat = re.search(r"(\d+\.\d+)", rebuilt_output)
    if mat:
        rebuilt_label = float(mat.group(1))

    mat = re.search(r"(\d+\.\d+)", origi_output)
    if mat:
        origi_label = float(mat.group(1))

    if abs(rebuilt_label - origi_label) < 0.01:
        pass_count += 1
        return True
    else:
        return False


if __name__ == '__main__':
    home_dir = os.path.abspath("./")
    data_dir = "/home/BTD-data"


    # TVM v0.7 O0
    print("TVM v0.7 O0 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v07_O0/efficientnet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v07_O0/inception_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v07_O0/mobilenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v07_O0/shufflenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v07_O0/resnet18_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/resnet18_tvm_O0/resnet18_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v07_O0/vgg16_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/vgg16_tvm_O0/vgg16_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O0 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O0/embedding_tvmO0_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding/embedding_tvm_O0"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # TVM v0.7 O3
    print("TVM v0.7 O3 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v07_O3/efficientnet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/efficientnet_tvm_O3/efficientnet_lite4_tvm_O3_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v07_O3/inception_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v07_O3/mobilenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/mobilenetv2_tvm_O3/mobilenetv2_7_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v07_O3/shufflenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/shufflenetv2_tvm_O3/shufflenetv2_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v07_O3/resnet18_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.7/resnet18_tvm_O3/resnet18_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v07_O3/vgg16_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "vgg16_tvm_O3/vgg16_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.7 O3 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O3/embedding_tvmO3_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding/embedding_tvm_O3"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # TVM v0.8 O0
    print("TVM v0.8 O0 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v08_O0/efficientnet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v08_O0/inception_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v08_O0/mobilenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v08_O0/shufflenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v08_O0/resnet18_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/resnet18_tvm_O0/resnet18_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v08_O0/vgg16_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/vgg16_tvm_O0/vgg16_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O0 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O0/embedding_tvmO0_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding_extra/embedding_tvm_v08_O0"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # TVM v0.8 O3
    print("TVM v0.8 O3 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v08_O3/efficientnet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/efficientnet_tvm_O3/efficientnet_lite4_tvm_O3_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v08_O3/inception_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v08_O3/mobilenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/mobilenetv2_tvm_O3/mobilenetv2_7_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v08_O3/shufflenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/shufflenetv2_tvm_O3/shufflenetv2_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v08_O3/resnet18_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/resnet18_tvm_O3/resnet18_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v08_O3/vgg16_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.8/vgg16_tvm_O3/vgg16_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.8 O3 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O3/embedding_tvmO3_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding_extra/embedding_tvm_v08_O3"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # TVM v0.9.dev O0
    print("TVM v0.9.dev O0 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v09_O0/efficientnet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v09_O0/inception_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v09_O0/mobilenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v09_O0/shufflenet_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v09_O0/resnet18_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/resnet18_tvm_O0/resnet18_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v09_O0/vgg16_tvm_O0_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/vgg16_tvm_O0/vgg16_tvm_O0_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O0 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O0/embedding_tvmO0_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding_extra/embedding_tvm_v09_O0"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # TVM v0.9.dev O3
    print("TVM v0.9.dev O3 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficient_tvm_v09_O3/efficientnet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/efficientnet_tvm_v09_O3/efficientnet_lite4_tvm_O3_strip"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_tvm_v09_O3/inception_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_tvm_v09_O3/mobilenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/mobilenetv2_tvm_v09_O3/mobilenetv2_7_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_tvm_v09_O3/shufflenet_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/shufflenetv2_tvm_v09_O3/shufflenetv2_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_tvm_v09_O3/resnet18_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/resnet18_tvm_O3/resnet18_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_tvm_v09_O3/vgg16_tvm_O3_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "TVM-v0.9.dev/vgg16_tvm_O3/vgg16_tvm_O3_strip"))
    print("Pass") if compare_tvm(o1, o2) else print("Failed")

    print("TVM v0.9.dev O3 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_tvm_v07_O3/embedding_tvmO3_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding_extra/embedding_tvm_v09_O3"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # Glow 2020
    print("Glow 2020 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficientnet_glow_2020/efficientnet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/efficientnet/efficientnet_lite4_strip.out"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_glow_2020/inceptionv1_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/inception_v1/inception_v1_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_glow_2020/mobilenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/mobilenet/mobilenetv2_7_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_glow_2020/shufflenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/shufflenet_v2/shufflenet_v2_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_glow_2020/resnet18_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/resnet18_glow/resnet18_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_glow_2020/vgg16_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2020/vgg16_glow/vgg16_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2020 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_glow_2020/embedding_glow_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding/embedding_glow"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # Glow 2021
    print("Glow 2021 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficientnet_glow_2021/efficientnet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/efficientnet/efficientnet_lite4_strip.out"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_glow_2021/inceptionv1_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/inception_v1/inception_v1_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_glow_2021/mobilenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/mobilenet/mobilenetv2_7_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_glow_2021/shufflenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/shufflenet_v2/shufflenet_v2_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_glow_2021/resnet18_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/resnet18_glow/resnet18_v1_7_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_glow_2021/vgg16_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2021/vgg16_glow/vgg16_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2021 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_glow_2021/embedding_glow_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding/embedding_glow"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")


    # Glow 2022
    print("Glow 2022 Efficientnet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/efficientnet_glow_2022/efficientnet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/efficientnet/efficientnet_lite4_strip.out"), os.path.join(home_dir, "cat_transpose.bin"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Inception")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/inception_glow_2022/inceptionv1_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/inception_v1/inception_v1_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Mobilenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/mobilenet_glow_2022/mobilenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/mobilenet/mobilenetv2_7_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Shufflenet")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/shufflenet_glow_2022/shufflenet_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/shufflenet_v2/shufflenet_v2_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Resnet18")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/resnet18_glow_2022/resnet18_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/resnet18_glow/resnet18_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Vgg16")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/vgg16_glow_2022/vgg16_glow_rebuild.py"))
    o2 = run_original_model(os.path.join(data_dir, "Glow-2022/vgg16_glow/vgg16_strip.out"))
    print("Pass") if compare_glow(o1, o2) else print("Failed")

    print("Glow 2022 Fasttext")
    o1 = run_rebuilt_model(os.path.join(home_dir, "evaluation/fasttext_embedding_glow_2022/embedding_glow_rebuild.py"), '')
    o2 = run_original_model(os.path.join(data_dir, "embedding/embedding_glow"), '')
    print("Pass") if compare_fasttext(o1, o2) else print("Failed")

    print("Overall: {}/63 models are correctly rebuilt.".format(pass_count))
