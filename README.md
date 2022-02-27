# BTD: DNN Executables Decompiler

Artifact for draft: "Decompiling x86 Deep Neural Network Executables"

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

### White-box Attack Results
https://www.dropbox.com/s/9fuxmfuaroqtvjm/whitebox_steal.zip?dl=0

### Recompilation
The first package includes recompiled new DNN executables on x86 platforms. This is in accordance with our recompilation evaluation in Section 6.5.  
https://www.dropbox.com/s/i8ub0kihusy1evk/cpu_recompile.zip?dl=0  

The second package includes legacy code migration demo. As clarified in Section 7 (the Discussion section; Downstream Applications paragraph), we decompiled x86 DNN executables, and try to migrate the decompiled models to GPUs by compiling them again using TVM with cuda as target device.  
https://www.dropbox.com/s/01zu0oyh00e57pw/gpu_recompile.zip?dl=0
