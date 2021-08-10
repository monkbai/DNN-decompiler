from PIL import Image
import numpy as np
import os
import torch
import json


def build_input_1():
    image_fn = "/home/lifter/Documents/tvm_output/cat.png"
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    with open(os.path.join('./', "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


def build_input_2():
    image_fn = "/home/lifter/Documents/tvm_output/cat.png"
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image) # - np.array([123.0, 117.0, 104.0])
        # image /= np.array([58.395, 57.12, 57.375])
        # image = image.transpose((2, 0, 1))
        # image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    with open(os.path.join('./', "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


# =================================================================


def build_one(img_path: str):
    image = Image.open(img_path).resize((224, 224))
    
    def transform_image(image):
        image = np.array(image)
        if image.ndim == 2:
            image = image.reshape(224, 224, 1)
            x = image
            image = np.append(image, image, axis=2)
            image = np.append(image, x, axis=2)
        print(np.array(image).shape)
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    bin_name = img_path[:img_path.rfind('.')] + '.bin'
    print(bin_name)
    with open(bin_name, "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


def build_one_nnfusion(img_path: str):
    image = Image.open(img_path).resize((224, 224))
    
    def transform_image(image):
        image = np.array(image)
        if image.ndim == 2:
            image = image.reshape(224, 224, 1)
            x = image
            image = np.append(image, image, axis=2)
            image = np.append(image, x, axis=2)
        print(np.array(image).shape)
        #image = np.array(image) - np.array([123.0, 117.0, 104.0])
        #image /= np.array([58.395, 57.12, 57.375])
        #image = image.transpose((2, 0, 1))
        #image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    #print("x", x.shape)
    #x = x.transpose(0, 3, 1, 2)
    print("x", x.shape)
    bin_name = img_path[:img_path.rfind('.')] + '.BIN'
    print(bin_name)
    with open(bin_name, "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


def main():
    img_dir = "/export/d1/zliudc/DLE_Decompiler/TVM/imagenet_part/"
    files = os.listdir(img_dir)
    for f in files:
        f = os.path.join(img_dir, f)
        if f.endswith('.JPEG'):
            print(f)
            # build_one(f)
            build_one_nnfusion(f)


def build_nlp_input(output_dir: str):
    output_dir = os.path.abspath(output_dir)

    for i in range(50):
        dummy_input = torch.randint(0, 10000, (7, 1))
        dummy_input = np.asarray(dummy_input)
        l = dummy_input.tolist()
        json_str = json.dumps(l)
        output_path = os.path.join(output_dir, '{}.txt'.format(i))
        with open(output_path, 'w') as f:
            f.write(json_str)
            f.close()


if __name__ == '__main__':
    # build_input_1()
    # build_input_2()
    # ===============
    main()
    # build_nlp_input("/export/d1/zliudc/DLE_Decompiler/TVM/embedding_input/")
