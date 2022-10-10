#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo "BTD home: $CUR_DIR"

# It takes roughly 22 hours to run

# ------- Glow 2020 -------
#       Decompilation
# Time: 2 and a half hours
# -------------------------
echo " - Decompiling efficientnet_glow_2020"
cd $CUR_DIR/evaluation/efficientnet_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/efficientnet
python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_glow_2020"
cd $CUR_DIR/evaluation/inception_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/inception_v1
python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_glow_2020"
cd $CUR_DIR/evaluation/mobilenet_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/mobilenet
python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_glow_2020"
cd $CUR_DIR/evaluation/shufflenet_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/shufflenet_v2
python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_glow_2020"
cd $CUR_DIR/evaluation/resnet18_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/resnet18_glow
python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_glow_ida $TMP_DIR/resnet18_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_glow_2020"
cd $CUR_DIR/evaluation/vgg16_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/vgg16_glow
python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_glow_ida $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_glow_2020"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2020/
python3 ./embedding_glow_decompile.py $DATA_DIR/embedding/embedding_glow_funcs $DATA_DIR/embedding/embedding_glow $DATA_DIR/embedding/label_glow.txt


# ------- Glow 2021 -------
#       Decompilation
# -------------------------
echo " - Decompiling efficientnet_glow_2021"
cd $CUR_DIR/evaluation/efficientnet_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/efficientnet
python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_glow_2021"
cd $CUR_DIR/evaluation/inception_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/inception_v1
python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_glow_2021"
cd $CUR_DIR/evaluation/mobilenet_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/mobilenet
python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_glow_2021"
cd $CUR_DIR/evaluation/shufflenet_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/shufflenet_v2
python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_glow_2021"
cd $CUR_DIR/evaluation/resnet18_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/resnet18_glow
python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_v1_7_funcs $TMP_DIR/resnet18_v1_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_glow_2021"
cd $CUR_DIR/evaluation/vgg16_glow_2021/
TMP_DIR=$DATA_DIR/Glow-2021/vgg16_glow
python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_glow_2021"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2021/
python3 ./embedding_glow_decompile.py $DATA_DIR/Glow-2021/embedding/embedding_2_funcs_backup $DATA_DIR/Glow-2021/embedding/embedding_2_strip.out $DATA_DIR/Glow-2021/embedding/label.txt


# ------- Glow 2022 -------
#       Decompilation
# -------------------------
echo " - Decompiling efficientnet_glow_2022"
cd $CUR_DIR/evaluation/efficientnet_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/efficientnet
python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_glow_2022"
cd $CUR_DIR/evaluation/inception_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/inception_v1
python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_glow_2022"
cd $CUR_DIR/evaluation/mobilenet_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/mobilenet
python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_glow_2022"
cd $CUR_DIR/evaluation/shufflenet_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/shufflenet_v2
python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_glow_2022"
cd $CUR_DIR/evaluation/resnet18_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/resnet18_glow
python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_glow_funcs $TMP_DIR/resnet18_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_glow_2022"
cd $CUR_DIR/evaluation/vgg16_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/vgg16_glow
python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_glow_2022"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2022/
python3 ./embedding_glow_decompile.py $DATA_DIR/embedding_extra/embedding_glow_2022_funcs $DATA_DIR/embedding_extra/embedding_glow_2022 $DATA_DIR/embedding_extra/label_glow_2022.txt


# ------- TVM v0.7 O0 -------
#        Decompilation
#         2 hours
# ---------------------------
echo " - Decompiling efficientnet_tvm_v07_O0"
cd $CUR_DIR/evaluation/efficient_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/efficientnet_tvm_O0
python3 ./efficient_tvm_O0_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v07_O0"
cd $CUR_DIR/evaluation/inception_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/inceptionv1_tvm_O0
python3 ./inception_tvm_O0_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v07_O0"
cd $CUR_DIR/evaluation/mobilenet_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/mobilenetv2_tvm_O0
python3 ./mobilenet_tvm_O0_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v07_O0"
cd $CUR_DIR/evaluation/shufflenet_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/shufflenetv2_tvm_O0
python3 ./shufflenet_tvm_O0_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v07_O0"
cd $CUR_DIR/evaluation/resnet18_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/resnet18_tvm_O0
python3 ./resnet18_tvm_O0_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v07_O0"
cd $CUR_DIR/evaluation/vgg16_tvm_v07_O0/
TMP_DIR=$DATA_DIR/TVM-v0.7/vgg16_tvm_O0
python3 ./vgg16_tvm_O0_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v07_O0"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v07_O0/
python3 ./embedding_tvmO0_decompile.py $DATA_DIR/embedding/embedding_tvm_O0_funcs $DATA_DIR/embedding/embedding_tvm_O0 $DATA_DIR/embedding/label_tvm_O0.txt


# ------- TVM v0.7 O3 -------
#        Decompilation
#          2 hours
# ---------------------------
echo " - Decompiling efficientnet_tvm_v07_O3"
cd $CUR_DIR/evaluation/efficient_tvm_v07_O3/
TMP_DIR=$DATA_DIR/TVM-v0.7/efficientnet_tvm_O3
python3 ./efficient_tvm_O3_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v07_O3"
cd $CUR_DIR/evaluation/inception_tvm_v07_O3/
TMP_DIR=$DATA_DIR/TVM-v0.7/inceptionv1_tvm_O3
python3 ./inception_tvm_O3_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v07_O3"
cd $CUR_DIR/evaluation/mobilenet_tvm_v07_O3/
TMP_DIR=$DATA_DIR/TVM-v0.7/mobilenetv2_tvm_O3
python3 ./mobilenet_tvm_O3_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v07_O3"
cd $CUR_DIR/evaluation/shufflenet_tvm_v07_O3/
TMP_DIR=$DATA_DIR/TVM-v0.7/shufflenetv2_tvm_O3
python3 ./shufflenet_tvm_O3_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v07_O3"
cd $CUR_DIR/evaluation/resnet18_tvm_v07_O3/
TMP_DIR=$DATA_DIR/TVM-v0.7/resnet18_tvm_O3
python3 ./resnet18_tvm_O3_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v07_O3"
cd $CUR_DIR/evaluation/vgg16_tvm_v07_O3/
TMP_DIR=$DATA_DIR/vgg16_tvm_O3
python3 ./vgg16_tvm_O3_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v07_O3"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v07_O3/
python3 ./embedding_tvmO3_decompile.py $DATA_DIR/embedding/embedding_tvm_O3_funcs $DATA_DIR/embedding/embedding_tvm_O3 $DATA_DIR/embedding/label_tvm_O3.txt


# ------- TVM v0.8 O0 -------
#        Decompilation
#          2.5 hours
# ---------------------------
echo " - Decompiling efficientnet_tvm_v08_O0"
cd $CUR_DIR/evaluation/efficient_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/efficientnet_tvm_O0
python3 ./efficient_tvm_O0_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v08_O0"
cd $CUR_DIR/evaluation/inception_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/inceptionv1_tvm_O0
python3 ./inception_tvm_O0_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v08_O0"
cd $CUR_DIR/evaluation/mobilenet_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/mobilenetv2_tvm_O0
python3 ./mobilenet_tvm_O0_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v08_O0"
cd $CUR_DIR/evaluation/shufflenet_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/shufflenetv2_tvm_O0
python3 ./shufflenet_tvm_O0_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v08_O0"
cd $CUR_DIR/evaluation/resnet18_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/resnet18_tvm_O0
python3 ./resnet18_tvm_O0_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v08_O0"
cd $CUR_DIR/evaluation/vgg16_tvm_v08_O0/
TMP_DIR=$DATA_DIR/TVM-v0.8/vgg16_tvm_O0
python3 ./vgg16_tvm_O0_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v08_O0"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v08_O0/
python3 ./embedding_tvmO0_decompile.py $DATA_DIR/embedding_extra/embedding_tvm_v08_O0_funcs $DATA_DIR/embedding_extra/embedding_tvm_v08_O0 $DATA_DIR/embedding_extra/label_tvm_v08_O0.txt


# ------- TVM v0.8 O3 -------
#        Decompilation
# ---------------------------
echo " - Decompiling efficientnet_tvm_v08_O3"
cd $CUR_DIR/evaluation/efficient_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/efficientnet_tvm_O3
python3 ./efficient_tvm_O3_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v08_O3"
cd $CUR_DIR/evaluation/inception_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/inceptionv1_tvm_O3
python3 ./inception_tvm_O3_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v08_O3"
cd $CUR_DIR/evaluation/mobilenet_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/mobilenetv2_tvm_O3
python3 ./mobilenet_tvm_O3_decompile.py $TMP_DIR/mobilenetv2_funcs $TMP_DIR/mobilenetv2_7_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v08_O3"
cd $CUR_DIR/evaluation/shufflenet_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/shufflenetv2_tvm_O3
python3 ./shufflenet_tvm_O3_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v08_O3"
cd $CUR_DIR/evaluation/resnet18_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/resnet18_tvm_O3
python3 ./resnet18_tvm_O3_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v08_O3"
cd $CUR_DIR/evaluation/vgg16_tvm_v08_O3/
TMP_DIR=$DATA_DIR/TVM-v0.8/vgg16_tvm_O3
python3 ./vgg16_tvm_O3_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v08_O3"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v08_O3/
python3 ./embedding_tvmO3_decompile.py $DATA_DIR/embedding_extra/embedding_tvm_v08_O3_funcs $DATA_DIR/embedding_extra/embedding_tvm_v08_O3 $DATA_DIR/embedding_extra/label_tvm_v08_O3.txt


# ------- TVM v0.9.dev O0 -------
#        Decompilation
#       
# ---------------------------
echo " - Decompiling efficientnet_tvm_v09_O0"
cd $CUR_DIR/evaluation/efficient_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/efficientnet_tvm_O0
python3 ./efficient_tvm_O0_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v09_O0"
cd $CUR_DIR/evaluation/inception_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/inceptionv1_tvm_O0
python3 ./inception_tvm_O0_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v09_O0"
cd $CUR_DIR/evaluation/mobilenet_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/mobilenetv2_tvm_O0
python3 ./mobilenet_tvm_O0_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v09_O0"
cd $CUR_DIR/evaluation/shufflenet_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/shufflenetv2_tvm_O0
python3 ./shufflenet_tvm_O0_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v09_O0"
cd $CUR_DIR/evaluation/resnet18_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/resnet18_tvm_O0
python3 ./resnet18_tvm_O0_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v09_O0"
cd $CUR_DIR/evaluation/vgg16_tvm_v09_O0/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/vgg16_tvm_O0
python3 ./vgg16_tvm_O0_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O0_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v09_O0"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v09_O0/
python3 ./embedding_tvmO0_decompile.py $DATA_DIR/embedding_extra/embedding_tvm_v09_O0_funcs $DATA_DIR/embedding_extra/embedding_tvm_v09_O0 $DATA_DIR/embedding_extra/label_tvm_v09_O0.txt


# ------- TVM v0.9.dev O3 -------
#        Decompilation
# ---------------------------
echo " - Decompiling efficientnet_tvm_v09_O3"
cd $CUR_DIR/evaluation/efficient_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/efficientnet_tvm_v09_O3
python3 ./efficient_tvm_O3_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling inception_tvm_v09_O3"
cd $CUR_DIR/evaluation/inception_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/inceptionv1_tvm_O3
python3 ./inception_tvm_O3_decompile.py $TMP_DIR/inceptionv1_funcs $TMP_DIR/inceptionv1_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling mobilenet_tvm_v09_O3"
cd $CUR_DIR/evaluation/mobilenet_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/mobilenetv2_tvm_v09_O3
python3 ./mobilenet_tvm_O3_decompile.py $TMP_DIR/mobilenetv2_funcs $TMP_DIR/mobilenetv2_7_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling shufflenet_tvm_v09_O3"
cd $CUR_DIR/evaluation/shufflenet_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/shufflenetv2_tvm_v09_O3
python3 ./shufflenet_tvm_O3_decompile.py $TMP_DIR/shufflenetv2_funcs $TMP_DIR/shufflenetv2_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling resnet18_tvm_v09_O3"
cd $CUR_DIR/evaluation/resnet18_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/resnet18_tvm_O3
python3 ./resnet18_tvm_O3_decompile.py $TMP_DIR/resnet18_funcs $TMP_DIR/resnet18_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_tvm_v09_O3"
cd $CUR_DIR/evaluation/vgg16_tvm_v09_O3/
TMP_DIR=$DATA_DIR/TVM-v0.9.dev/vgg16_tvm_O3
python3 ./vgg16_tvm_O3_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_tvm_O3_strip $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_tvm_v09_O3"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v09_O3/
python3 ./embedding_tvmO3_decompile.py $DATA_DIR/embedding_extra/embedding_tvm_v09_O3_funcs $DATA_DIR/embedding_extra/embedding_tvm_v09_O3 $DATA_DIR/embedding_extra/label_tvm_v09_O3.txt

echo "======= Decompilation Finished ======="
# =============
echo "======= Start Rebuild Evaluation ======="

# ------- Glow 2020 -------
#         Rebuild
# -------------------------
echo " - efficientnet_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficientnet_glow_2020/
python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/efficientnet/efficientnet_lite4_strip.out /home/cat_transpose.bin
echo ""

echo " - inception_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_glow_2020/
python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/inception_v1/inception_v1_strip.out /home/cat.bin
echo ""

echo " - mobilenet_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_glow_2020/
python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/mobilenet/mobilenetv2_7_strip.out /home/cat.bin
echo ""

echo " - shufflenet_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_glow_2020/
python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/shufflenet_v2/shufflenet_v2_strip.out /home/cat.bin
echo ""

echo " - resnet18_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_glow_2020/
python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/resnet18_glow/resnet18_strip.out /home/cat.bin
echo ""

echo " - vgg16_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_glow_2020/
python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2020/vgg16_glow/vgg16_strip.out /home/cat.bin
echo ""

echo " - fasttext_glow_2020"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2020/
python3 ./embedding_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding/embedding_glow
echo ""


# ------- Glow 2021 -------
#         Rebuild
# -------------------------
echo " - efficientnet_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficientnet_glow_2021/
python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/efficientnet/efficientnet_lite4_strip.out /home/cat_transpose.bin
echo ""

echo " - inception_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_glow_2021/
python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/inception_v1/inception_v1_strip.out /home/cat.bin
echo ""

echo " - mobilenet_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_glow_2021/
python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/mobilenet/mobilenetv2_7_strip.out /home/cat.bin
echo ""

echo " - shufflenet_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_glow_2021/
python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/shufflenet_v2/shufflenet_v2_strip.out /home/cat.bin
echo ""

echo " - resnet18_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_glow_2021/
python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/resnet18_glow/resnet18_v1_7_strip.out /home/cat.bin
echo ""

echo " - vgg16_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_glow_2021/
python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/vgg16_glow/vgg16_strip.out /home/cat.bin
echo ""

echo " - fasttext_glow_2021"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2021/
python3 ./embedding_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2021/embedding/embedding_2_strip.out
echo ""


# ------- Glow 2022 -------
#         Rebuild
# -------------------------
echo " - efficientnet_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficientnet_glow_2022/
python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/efficientnet/efficientnet_lite4_strip.out /home/cat_transpose.bin
echo ""

echo " - inception_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_glow_2022/
python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/inception_v1/inception_v1_strip.out /home/cat.bin
echo ""

echo " - mobilenet_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_glow_2022/
python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/mobilenet/mobilenetv2_7_strip.out /home/cat.bin
echo ""

echo " - shufflenet_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_glow_2022/
python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/shufflenet_v2/shufflenet_v2_strip.out /home/cat.bin
echo ""

echo " - resnet18_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_glow_2022/
python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/resnet18_glow/resnet18_strip.out /home/cat.bin
echo ""

echo " - vgg16_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_glow_2022/
python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/Glow-2022/vgg16_glow/vgg16_strip.out /home/cat.bin
echo ""

echo " - fasttext_glow_2022"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2022/
python3 ./embedding_glow_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_glow_2022
echo ""


# ------- TVM v0.7 O0 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v07_O0/
python3 ./efficientnet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip /home/cat_transpose.bin
echo ""

echo " - inception_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v07_O0/
python3 ./inception_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip /home/cat.bin
echo ""

echo " - mobilenet_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v07_O0/
python3 ./mobilenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip /home/cat.bin
echo ""

echo " - shufflenet_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v07_O0/
python3 ./shufflenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip /home/cat.bin
echo ""

echo " - resnet18_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v07_O0/
python3 ./resnet18_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/resnet18_tvm_O0/resnet18_tvm_O0_strip /home/cat.bin
echo ""

echo " - vgg16_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v07_O0/
python3 ./vgg16_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/vgg16_tvm_O0/vgg16_tvm_O0_strip /home/cat.bin
echo ""

echo " - fasttext_tvm_v07_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v07_O0/
python3 ./embedding_tvmO0_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding/embedding_tvm_O0
echo ""


# ------- TVM v0.7 O3 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v07_O3/
python3 ./efficientnet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/efficientnet_tvm_O3/efficientnet_lite4_tvm_O3_strip /home/cat_transpose.bin
echo ""

echo " - inception_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v07_O3/
python3 ./inception_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip /home/cat.bin
echo ""

echo " - mobilenet_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v07_O3/
python3 ./mobilenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/mobilenetv2_tvm_O3/mobilenetv2_7_tvm_O3_strip /home/cat.bin
echo ""

echo " - shufflenet_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v07_O3/
python3 ./shufflenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/shufflenetv2_tvm_O3/shufflenetv2_tvm_O3_strip /home/cat.bin
echo ""

echo " - resnet18_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v07_O3/
python3 ./resnet18_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.7/resnet18_tvm_O3/resnet18_tvm_O3_strip /home/cat.bin
echo ""

echo " - vgg16_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v07_O3/
python3 ./vgg16_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/vgg16_tvm_O3/vgg16_tvm_O3_strip /home/cat.bin
echo ""

echo " - fasttext_tvm_v07_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v07_O3/
python3 ./embedding_tvmO3_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding/embedding_tvm_O3
echo ""


# ------- TVM v0.8 O0 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v08_O0/
python3 ./efficientnet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip /home/cat_transpose.bin 2>&-
echo ""

echo " - inception_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v08_O0/
python3 ./inception_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - mobilenet_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v08_O0/
python3 ./mobilenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - shufflenet_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v08_O0/
python3 ./shufflenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - resnet18_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v08_O0/
python3 ./resnet18_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/resnet18_tvm_O0/resnet18_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - vgg16_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v08_O0/
python3 ./vgg16_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/vgg16_tvm_O0/vgg16_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - fasttext_tvm_v08_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v08_O0/
python3 ./embedding_tvmO0_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_tvm_v08_O0
echo ""


# ------- TVM v0.8 O3 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v08_O3/
python3 ./efficientnet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/efficientnet_tvm_O3/efficientnet_lite4_tvm_O3_strip /home/cat_transpose.bin
echo ""

echo " - inception_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v08_O3/
python3 ./inception_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip /home/cat.bin
echo ""

echo " - mobilenet_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v08_O3/
python3 ./mobilenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/mobilenetv2_tvm_O3/mobilenetv2_7_tvm_O3_strip /home/cat.bin
echo ""

echo " - shufflenet_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v08_O3/
python3 ./shufflenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/shufflenetv2_tvm_O3/shufflenetv2_tvm_O3_strip /home/cat.bin
echo ""

echo " - resnet18_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v08_O3/
python3 ./resnet18_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/resnet18_tvm_O3/resnet18_tvm_O3_strip /home/cat.bin
echo ""

echo " - vgg16_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v08_O3/
python3 ./vgg16_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.8/vgg16_tvm_O3/vgg16_tvm_O3_strip /home/cat.bin
echo ""

echo " - fasttext_tvm_v08_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v08_O3/
python3 ./embedding_tvmO3_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_tvm_v08_O3
echo ""


# ------- TVM v0.9.dev O0 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v09_O0/
python3 ./efficientnet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/efficientnet_tvm_O0/efficientnet_lite4_tvm_O0_strip /home/cat_transpose.bin 2>&-
echo ""

echo " - inception_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v09_O0/
python3 ./inception_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/inceptionv1_tvm_O0/inceptionv1_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - mobilenet_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v09_O0/
python3 ./mobilenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/mobilenetv2_tvm_O0/mobilenetv2_7_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - shufflenet_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v09_O0/
python3 ./shufflenet_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/shufflenetv2_tvm_O0/shufflenetv2_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - resnet18_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v09_O0/
python3 ./resnet18_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/resnet18_tvm_O0/resnet18_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - vgg16_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v09_O0/
python3 ./vgg16_tvm_O0_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/vgg16_tvm_O0/vgg16_tvm_O0_strip /home/cat.bin 2>&-
echo ""

echo " - fasttext_tvm_v09_O0"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v09_O0/
python3 ./embedding_tvmO0_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_tvm_v09_O0
echo ""


# ------- TVM v0.9.dev O3 -------
#           Rebuild
# ---------------------------
echo " - efficientnet_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/efficient_tvm_v09_O3/
python3 ./efficientnet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/efficientnet_tvm_v09_O3/efficientnet_lite4_tvm_O3_strip /home/cat_transpose.bin 2>&-
echo ""

echo " - inception_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/inception_tvm_v09_O3/
python3 ./inception_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/inceptionv1_tvm_O3/inceptionv1_tvm_O3_strip /home/cat.bin 2>&-
echo ""

echo " - mobilenet_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/mobilenet_tvm_v09_O3/
python3 ./mobilenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/mobilenetv2_tvm_v09_O3/mobilenetv2_7_tvm_O3_strip /home/cat.bin 2>&-
echo ""

echo " - shufflenet_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/shufflenet_tvm_v09_O3/
python3 ./shufflenet_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/shufflenetv2_tvm_v09_O3/shufflenetv2_tvm_O3_strip /home/cat.bin 2>&-
echo ""

echo " - resnet18_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/resnet18_tvm_v09_O3/
python3 ./resnet18_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/resnet18_tvm_O3/resnet18_tvm_O3_strip /home/cat.bin 2>&-
echo ""

echo " - vgg16_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/vgg16_tvm_v09_O3/
python3 ./vgg16_tvm_O3_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/TVM-v0.9.dev/vgg16_tvm_O3/vgg16_tvm_O3_strip /home/cat.bin 2>&-
echo ""

echo " - fasttext_tvm_v09_O3"
echo " - Rebuilt model output:"
cd $CUR_DIR/evaluation/fasttext_embedding_tvm_v09_O3/
python3 ./embedding_tvmO3_rebuild.py 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_tvm_v09_O3
echo ""
