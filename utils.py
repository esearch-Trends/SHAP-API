import os
import json
import shap
import pickle
import numpy as np
from PIL import Image
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing import image, image_dataset_from_directory

# function to get encodes
def number_to_string(argument: int):
    """
    function to get cardiac disease by encode value
    @param argument: encode value
    @return: cardiac disease
    """
    if argument == 0:
        return "Left Bundle Branch Block"
    elif argument == 1:
        return "Normal"
    elif argument == 2:
        return "Premature Atrial Contraction"
    elif argument == 3:
        return "Premature Ventricular Contractions"
    elif argument == 4:
        return "Right Bundle Branch Block"
    elif argument == 5:
        return "Ventricular Fibrillation"
    else:
        return "Unknown"

def preprocess_img(imgString: str):
    """"
    method to decode base64 image
    @param imgString: base64 image string
    @return: decoded image
    """
    decoded_image = imgString.convert('RGB')

    # Resize the image to the target size
    decoded_image = decoded_image.resize((64, 64))

    imgEncoded = image.img_to_array(decoded_image)
    processed_image = np.expand_dims(imgEncoded, axis=0)

    return processed_image

def shap_calculator(model, reference_image_path, folder_path):
    """"
    method to decode base64 image
    @param imgString: base64 image string
    @return: decoded image
    """
    try:
        # Load and preprocess the reference image
        reference_img = reference_image_path.convert('RGB')
        reference_img = reference_img.resize((64, 64))
        reference_img_array = np.array(reference_img)
        reference_img_array = reference_img_array.astype('float32') / 255.0

        # Load and preprocess the 6 images from the folder along with their filenames
        image_files = sorted(os.listdir(folder_path))[:6]
        images = []
        image_filenames = []
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize((64, 64))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            images.append(img_array)
            image_filenames.append(img_file)

        # Plot the images in a row with their filenames
        fig, axes = plt.subplots(1, 7, figsize=(16, 4))

        # Compute SHAP values for the reference image (replace 'model' with your model)
        background = np.zeros((1, 64, 64, 3))
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(np.array([reference_img_array]))

        shap_summary = []
        for values in shap_values:
            shap_mean = np.mean(np.abs(values))
            shap_summary.append(shap_mean)

        # Show the reference image separately
        axes[0].imshow(reference_img_array)
        axes[0].text(0.5, 1.15, 'Reference Image', transform=axes[0].transAxes,
                     ha='center', va='center', fontsize=10, color='black')
        axes[0].axis('off')

        # Add a border around the reference image
        rect_ref = patches.Rectangle((0, 0), 63, 63, linewidth=1, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect_ref)

        # Plot the 6 images from the folder with their respective filenames as titles and SHAP summary values
        for i, (img_array, filename, shap_val) in enumerate(zip(images, image_filenames, shap_summary)):
            axes[i+1].imshow(img_array)
            axes[i+1].axis('off')

            # Add a border around each image
            rect = patches.Rectangle((0, 0), 63, 63, linewidth=1, edgecolor='r', facecolor='none')
            axes[i+1].add_patch(rect)

            axes[i+1].text(0.5, -0.3, f'SHAP: {shap_val:.4f}', transform=axes[i+1].transAxes,
                           ha='center', va='center', fontsize=10, color='black')

            axes[i+1].text(0.5, 1.15, f'{filename[:-4]}', transform=axes[i+1].transAxes,
                           ha='center', va='center', fontsize=10, color='black')

        plt.tight_layout()

        plt.savefig('plot1.png', dpi=600)

        # Display the SHAP values plot
        shap.image_plot(shap_values, np.array([reference_img_array]), show=False)
        plt.savefig('plot2.png', dpi=600)
        plt.close()

        return True  # Both images saved successfully

    except Exception as e:
        print(f"Error: {e}")
        return False
