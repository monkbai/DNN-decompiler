from PIL import Image
import numpy as np
import os
    
    
def build_inputs():
    # Download test image
    #image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    #image_fn = os.path.join(build_dir, "cat.png")
    image_fn = './cat.png'
    #download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image)
        #image = np.array(image) - np.array([123.0, 117.0, 104.0])
        #image /= np.array([58.395, 57.12, 57.375])
        print(image.shape)
        # image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    with open(os.path.join('./', "cat1.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


def build2():
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
    
    image_path = './cat.png'
    
    x = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
    x = x[None, ...]
    x = preprocess_input(x)
    print(x.shape)
    with open(os.path.join('./', "cat2.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())
        

if __name__ == '__main__':
    build_inputs()
    build2()
