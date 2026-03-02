import string
import time
import numpy as np
import os
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical, get_file
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

from tqdm import tqdm
tqdm.pandas()

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# extract descriptions for images
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    img_captions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in img_captions:
            img_captions[img[:-2]] = [caption]
        else:
            img_captions[img[:-2]].append(caption)
    return img_captions

# clean descriptions
def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    
    for img, caps in captions.items():
        for i, img_caption in enumerate(caps):
            
            img_caption = img_caption.replace('-', ' ')  
            
            desc = img_caption.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1 and word.isalpha()]
            
            img_caption = ' '.join(desc)
            captions[img][i] = img_caption
            
    return captions

# create a list of all description strings
def text_vocab(descriptions):
    vocab = set()
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

# Get the folder where main.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute path 
images_dataset = os.path.join(BASE_DIR, 'dataset', 'Flicker8k_Dataset')

text_dataset = os.path.join(BASE_DIR, "dataset", "Flicker8k_text")

filename = os.path.join(text_dataset, "Flickr8k.token.txt")

# Debug print
print("Images folder exists:", os.path.exists(images_dataset)) 

descriptions= all_img_captions(filename)
print('Total Images: %d' % len(descriptions))

clean_descriptions = cleaning_text(descriptions)
vocab = text_vocab(clean_descriptions)
print('Vocabulary Size: %d' % len(vocab))
save_descriptions(clean_descriptions, 'descriptions.txt')

# load the pre-trained Xception model and remove the top layer to use it as a feature extractor
def download_with_retry(url, filename, retries=3):
    for attempt in range(retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == retries - 1:
                raise e
            print(f"Download attempt failed")
            time.sleep(5)  # Wait for 5 seconds before retrying
            
# Download the Xception model weights with retry logic
weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_path = download_with_retry(weights_url, "xception_weights.h5")
model = Xception(include_top=False, pooling='avg', weights=weights_path)

# extract features from all images and store them in a dictionary
def extract_features(directory, model):
    features = {}
    valid_images = (".jpg", ".jpeg", ".png")  # Use tuple for faster checking

    for img_name in tqdm(os.listdir(directory), desc="Extracting features"):
        if not img_name.lower().endswith(valid_images):
            continue

        img_path = os.path.join(directory, img_name)

        # Load and preprocess image
        image = load_img(img_path, target_size=(299, 299))  # 299x299 for Xception
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Extract features
        feature = model.predict(image, verbose=0)

        # Save features in dictionary using image name without extension
        features[img_name.split('.')[0]] = feature

    return features

feature = extract_features(images_dataset, model)
dump(feature, open("features.p", "wb"))


    