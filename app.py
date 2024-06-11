from flask import Flask, render_template, request
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from pathlib import Path
# import h5py

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

# Function to display resized image and predicted class
def display_images(model, class_name, filename):

    # Resize the image
    resize_img4_array = np.array(Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).resize((160, 160)))
    img4_array_batch=np.expand_dims(resize_img4_array,0)

    #prediction, class define, probability
    prediction = model.predict(img4_array_batch)
    class_ = class_name[np.argmax(prediction[0])]
    confidence = np.round(np.max(prediction[0])*100,2)

    return class_ , confidence

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        resized_img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).resize((360,360))
        resized_img.save(os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + filename))  # Save the resized image

        model_name = "xception"
        model_path = Path.cwd() / 'saved_model' / f'{model_name}' / f'{model_name}.h5'
        model = keras.models.load_model(model_path)
        # model = load_model(model_path)

        file_name = Path.cwd() / 'saved_model' / 'class_name.npy'
        class_name = np.load(file_name)
        
        # Call function to display resized image and predictions
        predicted_class, confidence = display_images(model, class_name, filename)
        
        return render_template('uploaded.html', filename=filename, 
                               predicted_class=predicted_class, confidence=confidence)


if __name__ == '__main__':
    app.run(debug=True)
