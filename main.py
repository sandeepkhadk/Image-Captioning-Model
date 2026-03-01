import string
import numpy as np
import os
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.xception import Xception, preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessimg.image import load_img, img_to_array
from keras.utils import to_categorical, get_file
from keras.models import Model,load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

# load doc into memory
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
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