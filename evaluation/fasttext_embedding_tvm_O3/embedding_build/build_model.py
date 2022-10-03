# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Creates a simple TVM modules."""

import argparse
import os
from tvm import relay
import tvm
from tvm import te
import logging
import json

RUNTIMES = {
    "c": "{name}_c.{ext}",
    "c++": "{name}_cpp.{ext}",
}


def build_module(opts):
    import onnx
    model_path = './embedding_2.onnx'
    onnx_model = onnx.load(model_path)
    input_name = "input.1"
    dshape = (7, 1)
    shape_dict = {input_name: dshape}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    mod = relay.transform.DynamicToStatic()(mod)
    
    # do we need the softmax at the end?
    func = mod["main"]
    """
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )
    """

    for runtime_name, file_format_str in RUNTIMES.items():
        with tvm.transform.PassContext(opt_level=0, config={"tir.disable_vectorize": True}):
            graph, lib, params = relay.build(
                func, f"llvm --runtime={runtime_name} --system-lib", params=params
            )

        build_dir = os.path.abspath(opts.out_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        lib.save(os.path.join(build_dir, file_format_str.format(name="model", ext="o")))
        with open(
            os.path.join(build_dir, file_format_str.format(name="graph", ext="json")), "w"
        ) as f_graph_json:
            f_graph_json.write(graph)
        with open(
            os.path.join(build_dir, file_format_str.format(name="params", ext="bin")), "wb"
        ) as f_params:
            f_params.write(relay.save_param_dict(params))


def build_inputs(opts):
    from tvm.contrib import download
    from PIL import Image
    import numpy as np

    build_dir = os.path.abspath(opts.out_dir)

    # Download test image
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join(build_dir, "cat.png")
    download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    with open(os.path.join(build_dir, "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default=".")
    parser.add_argument("-t", "--test", action="store_true")
    opts = parser.parse_args()

    if opts.test:
        build_test_module(opts)
    else:
        build_module(opts)
        build_inputs(opts)