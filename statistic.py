import os
import re
import sys

def scan(funcs_dir: str, f_count=0.0, a_count=0.0):
    func_count = f_count
    asm_count = a_count
    for filename in os.listdir(funcs_dir):
        if re.match(r"\d{4,4}\.", filename):
            func_count += 1
            f_path = os.path.join(funcs_dir, filename)
            if os.path.isfile(f_path):
                with open(f_path, 'r') as f:
                    asm_txt = f.read()
                    asm_count += len(asm_txt.split('\n'))
    return float(func_count), float(asm_count)


if __name__ == '__main__':
    data_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/"

    # TVM -O0
    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/vgg16_tvm_O0/vgg16_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/vgg16_tvm_O0/vgg16_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/vgg16_tvm_O0/vgg16_funcs"), func_count, asm_count)
    print("TVM -O0 VGG16:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/resnet18_tvm_O0/resnet18_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/resnet18_tvm_O0/resnet18_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/resnet18_tvm_O0/resnet18_funcs"), func_count, asm_count)
    print("TVM -O0 ResNet18:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/mobilenetv2_tvm_O0/mobilenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/mobilenetv2_tvm_O0/mobilenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/mobilenetv2_tvm_O0/mobilenet_funcs"), func_count, asm_count)
    print("TVM -O0 Mobilenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/efficientnet_tvm_O0/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/efficientnet_tvm_O0/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/efficientnet_tvm_O0/efficientnet_funcs"), func_count, asm_count)
    print("TVM -O0 Efficientnet:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/inceptionv1_tvm_O0/inceptionv1_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/inceptionv1_tvm_O0/inceptionv1_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/inceptionv1_tvm_O0/inceptionv1_funcs"), func_count, asm_count)
    print("TVM -O0 Inceptionv1:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/shufflenetv2_tvm_O0/shufflenetv2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/shufflenetv2_tvm_O0/shufflenetv2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/shufflenetv2_tvm_O0/shufflenetv2_funcs"), func_count, asm_count)
    print("TVM -O0 Shufflenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "embedding/embedding_tvm_O0_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "embedding_extra/embedding_tvm_v08_O0_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "embedding_extra/embedding_tvm_v09_O0_funcs"), func_count, asm_count)
    print("TVM -O0 Fasttext:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    # TVM -O3    
    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "vgg16_tvm_O3/vgg16_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/vgg16_tvm_O3/vgg16_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/vgg16_tvm_O3/vgg16_funcs"), func_count, asm_count)
    print("TVM -O3 VGG16:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/resnet18_tvm_O3/resnet18_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/resnet18_tvm_O3/resnet18_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/resnet18_tvm_O3/resnet18_funcs"), func_count, asm_count)
    print("TVM -O3 ResNet18:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/mobilenetv2_tvm_O3/mobilenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/mobilenetv2_tvm_O3/mobilenetv2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/mobilenetv2_tvm_v09_O3/mobilenetv2_funcs"), func_count, asm_count)
    print("TVM -O3 Mobilenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/efficientnet_tvm_O3/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/efficientnet_tvm_O3/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/efficientnet_tvm_v09_O3/efficientnet_funcs"), func_count, asm_count)
    print("TVM -O3 Efficientnet:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/inceptionv1_tvm_O3/inceptionv1_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/inceptionv1_tvm_O3/inceptionv1_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/inceptionv1_tvm_O3/inceptionv1_funcs"), func_count, asm_count)
    print("TVM -O3 Inceptionv1:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.7/shufflenetv2_tvm_O3/shufflenetv2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.8/shufflenetv2_tvm_O3/shufflenetv2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "TVM-v0.9.dev/shufflenetv2_tvm_v09_O3/shufflenetv2_funcs"), func_count, asm_count)
    print("TVM -O3 Shufflenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "embedding/embedding_tvm_O3_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "embedding_extra/embedding_tvm_v08_O3_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "embedding_extra/embedding_tvm_v09_O3_funcs"), func_count, asm_count)
    print("TVM -O3 Fasttext:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    # Glow
    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "vgg16_glow/vgg16_glow_ida"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/vgg16_glow/vgg16_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/vgg16_glow/vgg16_funcs"), func_count, asm_count)
    print("Glow VGG16:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2020/resnet18_glow/resnet18_glow_ida"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/resnet18_glow/resnet18_v1_7_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/resnet18_glow/resnet18_v1_7_funcs"), func_count, asm_count)
    print("Glow ResNet18:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2020/mobilenet/mobilenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/mobilenet/mobilenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/mobilenet/mobilenet_funcs"), func_count, asm_count)
    print("Glow Mobilenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2020/efficientnet/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/efficientnet/efficientnet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/efficientnet/efficientnet_funcs"), func_count, asm_count)
    print("Glow Efficientnet:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2020/inception_v1/inception_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/inception_v1/inception_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/inception_v1/inception_funcs"), func_count, asm_count)
    print("Glow Inceptionv1:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2020/shufflenet_v2/shufflenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/shufflenet_v2/shufflenet_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2022/shufflenet_v2/shufflenet_funcs"), func_count, asm_count)
    print("Glow Shufflenet v2:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count))

    func_count = asm_count = 0
    func_count, asm_count = scan(os.path.join(data_dir, "embedding/embedding_glow_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "Glow-2021/embedding/embedding_2_funcs"), func_count, asm_count)
    func_count, asm_count = scan(os.path.join(data_dir, "embedding_extra/embedding_glow_2022_funcs"), func_count, asm_count)
    print("Glow Fasttext:\nAvg. #Func {}, Avg. #Asm Inst {}\n".format(func_count/3, asm_count/3))
