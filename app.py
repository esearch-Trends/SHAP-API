# importing dependencies
import io
import os
import json
import shap
import pickle
import base64
import numpy as np
from PIL import Image
from datetime import date
from urllib.parse import unquote
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image, image_dataset_from_directory

# importing modules
from utils import number_to_string, preprocess_img,shap_calculator


# Iniatlize a Flask app
app = Flask('app')
CORS(app, resource={r"/api/*": {"origins": "*"}})
app.config['CORS HEADERS'] = 'content-Type'


# loading the model
model=load_model('ECG1.h5')
print("Model Loaded Successfully")


@app.route('/predictResult', methods=['POST'])
@cross_origin()
def price_predict():
    if request.method == 'POST':
        try:
            if request.args.get("baseImg"):

                base64_image_string = request.args.get("baseImg")
                print(f"Base64 Image: {base64_image_string}")

                base64_content = base64_image_string.split(",")[1]
                print(f"base64_content: {base64_content}")

                # decoded_image = Image.open(io.BytesIO(base64.b64decode(base64_image_string)))
                decoded_image = Image.open(io.BytesIO(base64.b64decode(base64_content)))

                pred = model.predict(preprocess_img(decoded_image))
                print(f"Predictions: {pred}")
                y_pred=np.argmax(pred)

                cardiac_string = number_to_string(y_pred)
                print(f"Prediction: {cardiac_string}")

                folder_path = 'images'

                result = shap_calculator(model, decoded_image, folder_path)

                if result:
                    with open('plot1.png', 'rb') as img_file1, open('plot2.png', 'rb') as img_file2:
                        img1_base64 = base64.b64encode(img_file1.read()).decode('utf-8')
                        img2_base64 = base64.b64encode(img_file2.read()).decode('utf-8')
                    return json.dumps(
                        {
                            "status": 200,
                            "message": "Prediction made successfully",
                            "condition": cardiac_string,
                            "base_img": img1_base64,
                            "pred_img": img2_base64
                        }
                    )
                else:
                    return json.dumps(
                        {
                            "status": 400,
                            "message": "Response failed at backend"
                        }
                    )    
            else:
                raise Exception("No argument provided")
        except Exception as e:
            error_message = f"Internal Server Error: {str(e)}"
            print(error_message)
            return json.dumps({
                "status": 500,
                "message": error_message
            })
    else:
        return "Cardiac route"


# Run the app
if __name__ == '__main__':
    app.run()
