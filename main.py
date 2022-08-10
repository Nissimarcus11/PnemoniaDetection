import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors

st.title('Pnemonia Detection')
# -------------------------------------------------------------
# Getting image from the site and saving
def save_upload(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0;


# -------------------------------------------------------------
# Extracting features of the uploaded image
def extract_features(img_path, model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocessed_image = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_image).flatten()
    normalized_result = result/norm(result)
    return normalized_result

# -------------------------------------------------------------
# loading saved models and features
filenames = pickle.load(open('filenames.pkl', 'rb'))
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))

pneumonia_filenames = pickle.load(open('pneumonia_filenames.pkl', 'rb'))
pneumonia_feature_list = np.array(pickle.load(open('pneumonia_embeddings.pkl', 'rb')))
# print(pneumonia_filenames[:5])

# -----------------------------------------------------------
# getting pre trained model and adding our layers

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) # standard size
model.trainable = False # model already trained, we just use the model
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# -----------------------------------------------------------
# getting getting image from user
uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    if save_upload(uploaded_file):
        #file uploaded successfully
        display_image = Image.open(uploaded_file)
        #feature extraction
        test_features = extract_features(os.path.join('uploads',uploaded_file.name),model)
        #process
        # -----------------------------------------------------------
        # KNN of normal images
        neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        neighbours.fit(feature_list)
        distances, indices = neighbours.kneighbors([test_features])
        values = distances.mean(axis=1)
        print(values)
        # -----------------------------------------------------------
        #KNN of effected neighbours
        effected_neighbours = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
        effected_neighbours.fit(pneumonia_feature_list)
        effected_distances, effected_indices = effected_neighbours.kneighbors([test_features])
        effected_values = effected_distances.mean(axis=1)
        print(effected_values)
        # -----------------------------------------------------------
        #displaying the result
        if (values[0] < 0.53 and effected_values[0]>0.5):
            st.write("The image is normal")
        elif (effected_values[0] > 0.6):
            st.write("Please upload a proper chest x-ray")
        else:
            if(values[0]<0.6):
                st.write("The patient have mild pneumonia")
            else:
                st.write("The patient have high pneumonia")

        st.image(display_image)
    else:
        st.header("Error uploading in file")