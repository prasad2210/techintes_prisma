from json import load
from logging.config import stopListening
import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2



model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img



vid = cv2.VideoCapture(0)
ret, frame = vid.read()
content_image = frame
content_image = cv2.resize(content_image, dsize=(content_image.shape[1], content_image.shape[0]), interpolation=cv2.INTER_CUBIC)
np_data = np.asarray(content_image)
np_final = np.expand_dims(np_data, axis=0)
np_final = tf.image.convert_image_dtype(np_final, tf.float32)
 

# print(np_final.shape)
# print(np_data.shape)



# trial = load_image('try.jpg')

# print(trial.shape)

# if np_final == trial:
#     print("prasad")



style_image = load_image('seated-nude.jpg')


# content_image = np.reshape(content_image, [1, content_image.shape[0], content_image.shape[1], content_image.shape[2]])


# style_image = cv2.imread('monet.jpeg')
# style_image = np.reshape(style_image, [1, style_image.shape[0], style_image.shape[1], style_image.shape[2]])

# plt.imshow(np.squeeze(style_image))
# plt.show()


stylized_image = model(tf.constant(np_final), tf.constant(style_image))[0]


stylized_image = np.squeeze(stylized_image)
print(type(stylized_image))
cv2.imshow('sdf',stylized_image)
cv2.waitKey(0) 
cv2.destroyAllWindows() 

# plt.imshow(np.squeeze(stylized_image))
# plt.show()


