#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo "BTD home: $CUR_DIR"

# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
# cd ..
# exit 0


# Glow 2020
# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/efficientnet
# python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# cd $CUR_DIR/evaluation/inception_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/inception_v1
# python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# cd $CUR_DIR/evaluation/mobilenet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/mobilenet
# python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# cd $CUR_DIR/evaluation/shufflenet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/shufflenet_v2
# python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# cd $CUR_DIR/evaluation/resnet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/resnet18_glow
# python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_glow_ida $TMP_DIR/resnet18_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

cd $CUR_DIR/evaluation/vgg16_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/vgg16_glow
python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_glow_ida $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/efficientnet
# python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt


# Rebuild
# Glow 2020
# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-

echo "inception_glow_2020"
cd $CUR_DIR/evaluation/inception_glow_2020/
python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-

echo "mobilenet_glow_2020"
cd $CUR_DIR/evaluation/mobilenet_glow_2020/
python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-

echo "shufflenet_glow_2020"
cd $CUR_DIR/evaluation/shufflenet_glow_2020/
python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-

echo "resnet_glow_2020"
cd $CUR_DIR/evaluation/resnet_glow_2020/
python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-

echo "vgg16_glow_2020"
cd $CUR_DIR/evaluation/vgg16_glow_2020/
python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
