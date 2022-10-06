#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo "BTD home: $CUR_DIR"

cd $CUR_DIR/evaluation/efficientnet_glow_2020/
python3 ./efficientnet_glow_rebuild.py 2>&-
cd ..
exit 0

cd $CUR_DIR/evaluation/efficientnet_glow_2020/
TMP_DIR=$DATA_DIR/Glow-2020/efficientnet
python3 ./efficientnet_glow_decompile.py $TMP_DIR/efficientnet_funcs $TMP_DIR/efficientnet_lite4_strip.out $TMP_DIR/cat.bin $TMP_DIR/func_call.log $TMP_DIR/ground_truth.txt
cd ..

cd $CUR_DIR/evaluation/efficientnet_glow_2020/
python3 ./efficientnet_glow_rebuild.py 2>&-
cd ..
