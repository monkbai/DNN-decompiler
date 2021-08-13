# BTD: DNN Executables Decompiler

Artifact for ASPLOS 2022 submission: "Decompiling x86 Deep Neural Network Executables"

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

# Recompilation
https://www.dropbox.com/s/i8ub0kihusy1evk/cpu_recompile.zip?dl=0  
https://www.dropbox.com/s/01zu0oyh00e57pw/gpu_recompile.zip?dl=0
