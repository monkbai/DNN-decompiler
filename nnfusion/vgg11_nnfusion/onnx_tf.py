import onnx
from onnx_tf.backend import prepare



model = onnx.load('vgg11_nnfusion_rebuild.onnx')
print('loaded')
tf_rep = prepare(model) 
print('prepared')
tf_rep.export_graph('vgg11_nnfusion_rebuild.pb')
print('erported')
exit(0)

import numpy as np
from IPython.display import display
from PIL import Image
print('Image 1:')
img = Image.open('two.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output))
print('Image 2:')
img = Image.open('three.png').resize((28, 28)).convert('L')
display(img)
output = tf_rep.run(np.asarray(img, dtype=np.float32)[np.newaxis, np.newaxis, :, :])
print('The digit is classified as ', np.argmax(output)) 