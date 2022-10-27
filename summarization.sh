#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo "\n======= Table 2 ======="
python3 ./statistics.py 

echo "\n======= Table 3 ======="
cd ./operator_inference
cp AE/* ./
python3 ./statistics.py 
cat ./acc.txt
cd ..

echo "\n======= Table 4 ======="
python3 ./parameter_accuracy.py

echo "\n======= Table 5 ======="
ccho "<Pass> means the model is 100% correctly rebuilt."
python3 ./recompile_correctness.py
