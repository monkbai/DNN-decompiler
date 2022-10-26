import os
import time

for (compiler, setting) in [
    ('GLOW', 'GLOW_2020'), ('GLOW', 'GLOW_2021'), ('GLOW', 'GLOW_2022'),
    ('TVM', 'TVM_v0.7_O0'), ('TVM', 'TVM_v0.8_O0'), ('TVM', 'TVM_v0.9.dev_O0'),
    ('TVM', 'TVM_v0.7_O3'), ('TVM', 'TVM_v0.8_O3'), ('TVM', 'TVM_v0.9.dev_O3')]:
    for model in ['resnet18', 'vgg16', 'inception_v1', 'shufflenet_v2', 'mobilenet', 'efficientnet']:
        os.system('python accuracy.py --compiler %s  --setting %s --model %s' % (compiler, setting, model))
        time.sleep(2)