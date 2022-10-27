#! /bin/bash
CUR_DIR=$PWD
DATA_DIR=/home/BTD-data

echo ""
echo "======= Table 2 ======="
python3 ./statistic.py 

echo ""
echo "======= Table 3 ======="
cd ./operator_inference
cp AE/* ./
python3 ./run_accuracy.py > /dev/null
cat ./acc.txt
cd ..

echo ""
echo "======= Table 4 ======="
python3 ./parameter_accuracy.py

echo ""
echo "======= Table 5 ======="
echo "<Pass> means the model is 100% correctly rebuilt."
python3 ./recompile_correctness.py
