#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import shap
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras.preprocessing import image_dataset_from_directory


# In[3]:


model=load_model('ECG1.h5')


# In[4]:


# Paths to the folder containing 6 images and the reference image
folder_path = 'images'
reference_image_path = 'test/VF.png'


# In[5]:


img=image.load_img(reference_image_path,target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
pred = model.predict(x)
y_pred=np.argmax(pred)


# In[8]:


def number_to_string(argument):
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

# Example usage:
y_pred = 2
result = number_to_string(y_pred)
print(result)


# In[9]:


# Load and preprocess the reference image
reference_img = Image.open(reference_image_path)
reference_img = reference_img.convert('RGB')  # Convert to RGB mode if needed
reference_img = reference_img.resize((64, 64))  # Resize as necessary
reference_img_array = np.array(reference_img)  # Convert PIL image to NumPy array
reference_img_array = reference_img_array.astype('float32') / 255.0  # Normalize pixel values between 0 and 1

# Load and preprocess the 6 images from the folder along with their filenames
image_files = sorted(os.listdir(folder_path))[:6]  # Assuming the first 6 files
images = []
image_filenames = []
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    img = Image.open(img_path)
    img = img.convert('RGB')  # Convert to RGB mode if needed
    img = img.resize((64, 64))  # Resize as necessary
    img_array = np.array(img)  # Convert PIL image to NumPy array
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values between 0 and 1
    images.append(img_array)
    image_filenames.append(img_file)  # Store image filenames

# Plot the images in a row with their filenames
fig, axes = plt.subplots(1, 7, figsize=(16, 4))


# Compute SHAP values for the reference image (replace 'model' with your model)
background = np.zeros((1, 64, 64, 3))  # Create a background of zeros (single black image)
e = shap.DeepExplainer(model, background)
shap_values = e.shap_values(np.array([reference_img_array]))

shap_summary = []
for values in shap_values:
    shap_mean = np.mean(np.abs(values))
    shap_summary.append(shap_mean) 

# Show the reference image separately
axes[0].imshow(reference_img_array)
# axes[0].set_title('Reference Image')
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

# Save the plot
plt.savefig('plot1.png', dpi=600)

# Display the SHAP values plot
shap.image_plot(shap_values, np.array([reference_img_array]), show=False)
plt.savefig('plot2.png', dpi=600)
plt.show()


# In[ ]:




