import numpy as np
import tvm
from tvm import relay
from tvm.relay import testing
from tvm.contrib import graph_runtime
import onnx

import sys

## load mxnet model
if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    model_path = './vgg16_tvmO0_rebuild.onnx'

onnx_model = onnx.load(model_path)
input_name = "input"
dshape = (1, 3, 224, 224)
shape_dict = {input_name: dshape}
relay_func, relay_params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
relay_func = relay.transform.DynamicToStatic()(relay_func)

# set target as GPU, build TVM moduel
## ---------------------------- 
# graph：execution graph in json format
# lib: tvm module library of compiled functions for the graph on the target hardware
# params: parameter blobs
## ---------------------------
target = 'cuda'
with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(relay_func, target, params=relay_params)

# run forward
## use the cat png we preprocessed
with open("/export/d1/zliudc/DLE_Decompiler/TVM/rebuild_ida/vgg16_tvm_O0/cat.bin", 'br') as f:
    bin_data = f.read()
    np_arr = np.frombuffer(bin_data, dtype=np.float32)
    print(np_arr.shape)
    np_arr = np_arr.reshape(3, 224, 224)
    np_arr = np_arr.reshape((1, 3, 224, 224))
    x = np_arr
    print(x.shape)

# let's go
ctx = tvm.gpu(0)
dtype = 'float32'
## load the module
m = graph_runtime.create(graph, lib, ctx)
## set input data
m.set_input('input', tvm.nd.array(x.astype(dtype)))
## set input params
m.set_input(**params)
m.run()
# get output
outputs = m.get_output(0)
print(outputs.shape)
print(type(outputs))
top1 = np.argmax(outputs.asnumpy()[0])
print(top1)
print(outputs.asnumpy()[0,top1])

exit(0)

# save model
path_lib = './deploy_lib.tar'
lib.export_library(path_lib)

## store the computational graph as json file
with open('./deploy_graph.json', 'w') as f:
    f.write(graph)
## store weights as binary file 
with open('./deploy_params', 'wb') as f:
    f.write(relay.save_param_dict(params))

# load model back
loaded_json = open('./deploy_graph.json').read()
loaded_lib = tvm.module.load(path_lib)
loaded_params = bytearray(open('./deploy_params', 'rb').read())
module = graph_runtime.create(loaded_json, loaded_lib, ctx)

