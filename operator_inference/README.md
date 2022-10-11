# Operator Inference of BTD

## Structure

This folder is organized as:

```
📂operator_inference
 ┣ 📂data
 ┃ ┣ 📂bpe
 ┃ ┣ 📂dataset
 ┃ ┣ 📂index
 ┃ ┣ 📂label
 ┃ ┗ 📂vocab
 ┣ 📂output
 ┃ ┣ 📂TVM_v0.7_O0
 ┃ ┣ 📂TVM_v0.7_O3
 ┃ ┣ 📂TVM_v0.8_O0
 ┃ ┣ 📂TVM_v0.8_O3
 ┃ ┣ 📂TVM_v0.9_O0
 ┃ ┣ 📂TVM_v0.9_O3
 ┃ ┣ 📂GLOW_2020
 ┃ ┣ 📂GLOW_2021
 ┃ ┗ 📂GLOW_2022
 ┗ 📜*.py
```

`data` contains all the data for training the operator identifier.

`output` contains checkpoints of the trained operator identifier.

`*.py` are all scripts for 1) preparing the data, 2) training operator identifier, and 3) inferring DNN operators.

## BPE

Use [fastBPE](https://github.com/glample/fastBPE) to get the BPE code. We also prepare generated BPE codes [here](https://www.dropbox.com/s/prg0vmei2x781wy/data.zip?dl=0).

## Pre-processing

We provide all pre-processing functions in `preprocessing.py`. You can use these
functions to generate the training and test samples from [raw data](https://www.dropbox.com/s/a1mxqwqn4tytmgz/labeled_dataset_2022.zip?dl=0) provided in **Labeled Dataset for Operator Inference**. 

We also provide all the processed data at [here](https://www.dropbox.com/s/prg0vmei2x781wy/data.zip?dl=0). You can download and unzip them into the `data` folder.

## Training

*Note: you may need to replace `python3` with `python` in the following
instructions if you have both python 2.x and python 3.x installed.*

To train an operator identifier from scratch, run
`python main.py --training 1` with configurations explained below.

- `--compiler` - The compiler that compiles the executable. Choices are `['TVM', 'GLOW']`.  
- `--setting` - The corresponding compiler and optimization level of the exetuable. Choices are
`['TVM_v0.7_O0', 'TVM_v0.8_O0', 'TVM_v0.9.dev_O0', 'TVM_v0.7_O3', 'TVM_v0.8_O3', 'TVM_v0.9.dev_O3',
'GLOW_2020', 'GLOW_2021', 'GLOW_2022']`.

## Inference

To infer DNN operators using a trained operator identifier, run
`python main.py --training 0` with configurations lised in **Training**.

We also provide all our trained operator identifiers [here](https://www.dropbox.com/s/e8rgxp2u3f01omn/output.zip?dl=0). You can download and unzip them into the `output` folder.

To infer DNN operators using these checkpoints,
run `python main.py --training 0` and set `--exp_name` as one of the choices from
`['TVM_v0.7_O0', 'TVM_v0.8_O0', 'TVM_v0.9.dev_O0', 'TVM_v0.7_O3', 'TVM_v0.8_O3', 'TVM_v0.9.dev_O3',
'GLOW_2020', 'GLOW_2021', 'GLOW_2022']`.

For example, if you want to infer DNN operators for an executable compiled using TVM ver. 0.7 and O0 optimization, you can run `python main.py --training 0 --exp_name TVM_v0.7_O0 --setting TVM_v0.7_O0 --compiler TVM`. Similarly, for executables compiled using GLOW ver. 2020, you can
run `python main.py --training 0 --exp_name GLOW_2020 --setting GLOW_2020 --compiler GLOW`.
