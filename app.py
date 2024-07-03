import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
model =load_model('/Users/Krishna/Desktop/New copy/model/model.h5')


print('Model loaded. Check http://127.0.0.1:5000')


def get_className(prediction):
    if prediction==0:
        return "Glioma Brain Tumor Detected"
    elif prediction==1:
        return "Meningioma Brain Tumor Detected"
    elif prediction==2:
        return "No Brain Tumor Detected"
    else:
        return "Pituitary Brain Tumor Detected"



def predict(img):
    test_image=load_img(img,target_size=(150,150))
    test_image.resize((150,150))
    image = img_to_array(test_image)
    image = image / 255.0
    image = np.expand_dims(image, axis = 0)
    result = np.argmax(model.predict(image),axis=1)
    return result



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# @app.route('/model', methods=['GET'])
# def model_page():
#     return render_template('model_playground.html')

# @app.route('/contact', methods=['GET'])
# def contact():
#     return render_template('home.html#contact')

@app.route('/submit', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        #request.form
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=predict(file_path)
        r=get_className(value) 
        return r
        
    return None #render_template('index.html',result=r)


if __name__ == '__main__':
    app.run(debug=True)