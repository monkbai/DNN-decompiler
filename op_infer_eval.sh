#! /bin/bash
CUR_DIR=$PWD

cd ./operator_inference

# Run the operator inference evaluation with pre-trained model
python3 main.py --training 0 --exp_name GLOW_2020 --setting GLOW_2020 --compiler GLOW
echo "----------------------------------------"
python3 main.py --training 0 --exp_name GLOW_2021 --setting GLOW_2021 --compiler GLOW
echo "----------------------------------------"
python3 main.py --training 0 --exp_name GLOW_2022 --setting GLOW_2022 --compiler GLOW
echo "----------------------------------------"

python3 main.py --training 0 --exp_name TVM_v0.7_O0 --setting TVM_v0.7_O0 --compiler TVM
echo "----------------------------------------"
python3 main.py --training 0 --exp_name TVM_v0.7_O3 --setting TVM_v0.7_O3 --compiler TVM
echo "----------------------------------------"
python3 main.py --training 0 --exp_name TVM_v0.8_O0 --setting TVM_v0.8_O0 --compiler TVM
echo "----------------------------------------"
python3 main.py --training 0 --exp_name TVM_v0.8_O3 --setting TVM_v0.8_O3 --compiler TVM
echo "----------------------------------------"
python3 main.py --training 0 --exp_name TVM_v0.9.dev_O0 --setting TVM_v0.9.dev_O0 --compiler TVM
echo "----------------------------------------"
python3 main.py --training 0 --exp_name TVM_v0.9.dev_O3 --setting TVM_v0.9.dev_O3 --compiler TVM
echo "----------------------------------------"

echo "Inference results are written in $CUR_DIR/operator_inference/output/<compiler_option>/text/test_000.txt>"
