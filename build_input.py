from PIL import Image
import numpy as np
import os


def build_inputs():
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


if __name__ == '__main__':
    # build_inputs()
    build_input_2()
