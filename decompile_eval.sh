#! /bin/bash
CUR_DIR=$PWD

echo "BTD home: $CUR_DIR"

cd $CUR_DIR/evaluation/efficientnet_glow_2020/
python3 ./efficientnet_glow_rebuild.py
cd ..
