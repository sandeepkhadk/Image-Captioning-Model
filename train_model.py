import string
import numpy as np
import os
from pickle import dump, dumps, load
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tqdm import tqdm
from feature_extraction import load_doc, all_img_captions, cleaning_text, text_vocab

tqdm.pandas()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

images_dataset = os.path.join(BASE_DIR, 'dataset', 'Flicker8k_Dataset')
text_dataset = os.path.join(BASE_DIR, "dataset", "Flicker8k_text")

# load features
features = load(open("features.p", "rb"))

print("Features loaded successfully")


def load_photos(filename):
    file = load_doc(filename)
    photos = file.split('\n')[:-1]
    photos_present = [
        photo for photo in photos
        if os.path.exists(os.path.join(images_dataset, photo))
    ]
    return photos_present


def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}

    for line in file.split('\n'):

        tokens = line.split('\t')

        if len(tokens) < 1:
            continue

        image_id, image_desc = tokens[0], tokens[1:]

        if image_id in photos:

            if image_id not in descriptions:
                descriptions[image_id] = []

            desc = '<start> ' + ' '.join(image_desc) + ' <end>'

            descriptions[image_id].append(desc)

    return descriptions


def load_photo_features(filename, photos):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in photos if k in all_features}
    print(f"Loaded features for {len(features)} photos.")
    return features


filename = text_dataset + "/Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)

train_descriptions = load_clean_descriptions('descriptions.txt', train_imgs)

train_features = load_photo_features("features.p", train_imgs)


def dict_to_list(descriptions):
    all_desc = []

    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]

    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)

# Save tokenizer
with open("tokenizer.p", "wb") as f:
    dump(tokenizer, f)

print("Tokenizer saved successfully")