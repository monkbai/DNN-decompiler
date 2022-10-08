#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo "BTD home: $CUR_DIR"



# ------- Glow 2020 -------
#       Decompilation
# -------------------------
# echo " - Decompiling efficientnet_glow_2020"
# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/efficientnet
# python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling inception_glow_2020"
# cd $CUR_DIR/evaluation/inception_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/inception_v1
# python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling mobilenet_glow_2020"
# cd $CUR_DIR/evaluation/mobilenet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/mobilenet
# python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling shufflenet_glow_2020"
# cd $CUR_DIR/evaluation/shufflenet_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/shufflenet_v2
# python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling resnet18_glow_2020"
# cd $CUR_DIR/evaluation/resnet18_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/resnet18_glow
# python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_glow_ida $TMP_DIR/resnet18_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling vgg16_glow_2020"
# cd $CUR_DIR/evaluation/vgg16_glow_2020/
# TMP_DIR=$DATA_DIR/Glow-2020/vgg16_glow
# python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_glow_ida $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling fasttext_glow_2020"
# cd $CUR_DIR/evaluation/fasttext_embedding_glow_2020/
# python3 ./embedding_glow_decompile.py $DATA_DIR/embedding/embedding_glow_funcs $DATA_DIR/embedding/embedding_glow $DATA_DIR/embedding/label_glow.txt


# ------- Glow 2021 -------
#       Decompilation
# -------------------------
# echo " - Decompiling efficientnet_glow_2021"
# cd $CUR_DIR/evaluation/efficientnet_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/efficientnet
# python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling inception_glow_2021"
# cd $CUR_DIR/evaluation/inception_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/inception_v1
# python3 ./inception_glow_decompile.py $TMP_DIR/inception_funcs $TMP_DIR/inception_v1_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling mobilenet_glow_2021"
# cd $CUR_DIR/evaluation/mobilenet_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/mobilenet
# python3 ./mobilenet_glow_decompile.py $TMP_DIR/mobilenet_funcs $TMP_DIR/mobilenetv2_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling shufflenet_glow_2021"
# cd $CUR_DIR/evaluation/shufflenet_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/shufflenet_v2
# python3 ./shufflenet_glow_decompile.py $TMP_DIR/shufflenet_funcs $TMP_DIR/shufflenet_v2_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling resnet18_glow_2021"
# cd $CUR_DIR/evaluation/resnet18_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/resnet18_glow
# python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_v1_7_funcs $TMP_DIR/resnet18_v1_7_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling vgg16_glow_2021"
# cd $CUR_DIR/evaluation/vgg16_glow_2021/
# TMP_DIR=$DATA_DIR/Glow-2021/vgg16_glow
# python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

# echo " - Decompiling fasttext_glow_2021"
# cd $CUR_DIR/evaluation/fasttext_embedding_glow_2021/
# python3 ./embedding_glow_decompile.py $DATA_DIR/Glow-2021/embedding/embedding_2_funcs_backup $DATA_DIR/Glow-2021/embedding/embedding_2_strip.out $DATA_DIR/Glow-2021/embedding/label.txt


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
python3 ./resnet18_glow_decompile.py $TMP_DIR/resnet18_v1_7_funcs $TMP_DIR/resnet18_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling vgg16_glow_2022"
cd $CUR_DIR/evaluation/vgg16_glow_2022/
TMP_DIR=$DATA_DIR/Glow-2022/vgg16_glow
python3 ./vgg16_glow_decompile.py $TMP_DIR/vgg16_funcs $TMP_DIR/vgg16_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/label.txt

echo " - Decompiling fasttext_glow_2022"
cd $CUR_DIR/evaluation/fasttext_embedding_glow_2022/
python3 ./embedding_glow_decompile.py $DATA_DIR/embedding_extra/embedding_glow_2022_funcs $DATA_DIR/embedding_extra/embedding_glow_2022 $DATA_DIR/embedding_extra/label_glow_2022.txt



# ------- Glow 2020 -------
#         Rebuild
# -------------------------
# echo " - efficientnet_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/efficientnet_glow_2020/
# python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/efficientnet/efficientnet_lite4_strip.out /home/cat_transpose.bin
# echo ""

# echo " - inception_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/inception_glow_2020/
# python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/inception_v1/inception_v1_strip.out /home/cat.bin
# echo ""

# echo " - mobilenet_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/mobilenet_glow_2020/
# python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/mobilenet/mobilenetv2_7_strip.out /home/cat.bin
# echo ""

# echo " - shufflenet_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/shufflenet_glow_2020/
# python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/shufflenet_v2/shufflenet_v2_strip.out /home/cat.bin
# echo ""

# echo " - resnet18_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/resnet18_glow_2020/
# python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/resnet18_glow/resnet18_strip.out /home/cat.bin
# echo ""

# echo " - vgg16_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/vgg16_glow_2020/
# python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2020/vgg16_glow/vgg16_strip.out /home/cat.bin
# echo ""

# echo " - fasttext_glow_2020"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/fasttext_embedding_glow_2020/
# python3 ./embedding_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/embedding/embedding_glow
# echo ""


# ------- Glow 2021 -------
#         Rebuild
# -------------------------
# echo " - efficientnet_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/efficientnet_glow_2021/
# python3 ./efficientnet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/efficientnet/efficientnet_lite4_strip.out /home/cat_transpose.bin
# echo ""

# echo " - inception_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/inception_glow_2021/
# python3 ./inceptionv1_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/inception_v1/inception_v1_strip.out /home/cat.bin
# echo ""

# echo " - mobilenet_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/mobilenet_glow_2021/
# python3 ./mobilenet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/mobilenet/mobilenetv2_7_strip.out /home/cat.bin
# echo ""

# echo " - shufflenet_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/shufflenet_glow_2021/
# python3 ./shufflenet_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/shufflenet_v2/shufflenet_v2_strip.out /home/cat.bin
# echo ""

# echo " - resnet18_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/resnet18_glow_2021/
# python3 ./resnet18_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/resnet18_glow/resnet18_v1_7_strip.out /home/cat.bin
# echo ""

# echo " - vgg16_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/vgg16_glow_2021/
# python3 ./vgg16_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/vgg16_glow/vgg16_strip.out /home/cat.bin
# echo ""

# echo " - fasttext_glow_2021"
# echo " - Rebuilt model output:"
# cd $CUR_DIR/evaluation/fasttext_embedding_glow_2021/
# python3 ./embedding_glow_rebuild.py /home/cat.bin 2>&-
# echo " - DNN Executable output:"
# $DATA_DIR/Glow-2021/embedding/embedding_2_strip.out
# echo ""


# ------- Glow 2022 -------
#         Rebuild
-------------------------
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
python3 ./embedding_glow_rebuild.py /home/cat.bin 2>&-
echo " - DNN Executable output:"
$DATA_DIR/embedding_extra/embedding_glow_2022
echo ""
