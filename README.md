# BTD: DNN Executables Decompiler

Research Artifact for our USENIX Security 2023 paper: "Decompiling x86 Deep Neural Network Executables"

BTD is the **first** deep neural network (DNN) executables decompiler. BTD takes DNN executables (running on x86 CPUs) and outputs full model specifications, including types of DNN operators, network topology, dimensions, and parameters that are (nearly) identical to those of the input models. BTD is evaluated to be robust against complex compiler optimizations, such as operator fusion and memory layout optimization. More details are reported in our paper published at USENIX Security 2023.

Paper: [will release soon](README.md)

Extended version: [https://arxiv.org/abs/2210.01075](https://arxiv.org/abs/2210.01075)

TODO: We will release and update all code and data in a few days and a usable Docker image will be available for artifact evaluation at that time. Please check this repo later.

## Prerequisites
```
ubuntu 18.04
git
gcc/g++ (7.5.0)
make (4.1)
python3 (3.6.9 or higher)
numpy-1.19.5
torch (1.9.0 or higher)
Intel pin (3.14) 
IDA Pro (optional)
```
You can download pin 3.14 from [here](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html), or use the docker image with all prerequisites installed.

BTD relies on IDA Pro (version 7.5) for disassembly, and because IDA is commercial software, we do not provide it in this repo; instead, in order to reduce the workload of AE reviewers, we provide the disassembly results directly as input for BTD. The scripts used to disassemble DNN executable into assembly functions with IDA are presented in [ida/](https://github.com/monkbai/DNN-decompiler/tree/master/ida). IDA Pro is not indispensable; any other full-fledged disassembly tool can be used to replace IDA, but we do not provide the relevant code here.

## 0.Prepare

Download and unzip Intel pin 3.14, then update the pin home directory in [config.py](https://github.com/monkbai/DNN-decompiler/blob/master/config.py#L3).

```
git clone https://github.com/monkbai/DNN-decompiler.git
mkdir <path_to_pin_home>/source/tools/MyPinTool/obj-intel64
cd DNN-decompiler
python3 pin_tool.py
```
[pin_tool.py](https://github.com/monkbai/DNN-decompiler/blob/master/pin_tools.py#L101) will copy and compile all pin tools listed in [MyPinTool/](https://github.com/monkbai/DNN-decompiler/tree/master/MyPinTool).

## Structure

### Pin tools
MyPinTool/

version: pin-3.14

### Operator Inference
operator_inference/

### Symbolic Engine
se_engine.py

### Taint Analysis
trace_filter.py

### Disassembly Scripts
ida/

### Decompile&Rebuild Scripts
resnet18_\*/  

vgg16_\*/  

embedding_\*/

nnfusion\*/

op_comprehensive/\*

### Appendix of Our Paper
docs/

## Data

### Labeled Dataset for Operator Inference
https://www.dropbox.com/s/lgkp2xfmyn7kwv4/labeled_dataset.zip?dl=0

### ONNX Models
https://www.dropbox.com/s/x8gwqyej7fla3rz/DNN_models.zip?dl=0

### Compiled DNN Executables
https://www.dropbox.com/s/lf297rjgx7e39id/DNN_executables.zip?dl=0

### Model Inputs
https://www.dropbox.com/s/nook5hs9srjveev/images.zip?dl=0  
https://www.dropbox.com/s/9y0k71dbowixs8w/embedding_input.zip?dl=0

### White-box Attack Results
https://www.dropbox.com/s/9fuxmfuaroqtvjm/whitebox_steal.zip?dl=0

### Recompilation
The first package includes recompiled new DNN executables on x86 platforms. This is in accordance with our recompilation evaluation in Section 6.5.  
https://www.dropbox.com/s/i8ub0kihusy1evk/cpu_recompile.zip?dl=0  

The second package includes legacy code migration demo. As clarified in Section 7 (the Discussion section; Downstream Applications paragraph), we decompiled x86 DNN executables, and try to migrate the decompiled models to GPUs by compiling them again using TVM with cuda as target device.  
https://www.dropbox.com/s/01zu0oyh00e57pw/gpu_recompile.zip?dl=0
