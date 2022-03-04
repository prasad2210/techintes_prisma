import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2

from flask import Flask, escape, request, render_template, Response, make_response, redirect

model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

vid = cv2.VideoCapture(0)

def filter(path):
    style_image = cv2.imread(path)
    style_image = np.expand_dims(style_image, axis=0)
    style_image = tf.image.convert_image_dtype(style_image, tf.float32)
    while True:
        ret, frame = vid.read()
        
        np_final = np.expand_dims(frame, axis=0)
        np_final = tf.image.convert_image_dtype(np_final, tf.float32)


        stylized_image = model(tf.constant(np_final), tf.constant(style_image))[0]
        stylized_image=np.array(stylized_image)
        
        stylized_image = np.squeeze(stylized_image)
        stylized_image = stylized_image*255
        stylized_image = np.uint8(stylized_image)
        
        # print(stylized_image)
        
        ret2, buffer2 = cv2.imencode('.JPEG', stylized_image)
        frame = buffer2.tobytes()

        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


app = Flask(__name__)

path = ['starrynight.jfif']

@app.route('/', methods=['GET', 'POST'])
def index():
    
    if request.method == "POST":
        path[0] = request.values.get('img1')
        
        print(path)
        
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(filter(path[0]), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.debug = True
    app.run()
    
    
