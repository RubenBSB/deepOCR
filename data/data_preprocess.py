import numpy as np
import cv2
import glob
import re

root = "data/words/*/*/*.png"
filenames = sorted(glob.glob(root))

def get_images():
    global filenames
    X = []
    for file in filenames:
        img = cv2.imread(file,0)
        X.append(img)
    return X

def get_words():
    pattern = "[^ ]+$"
    words =[]
    with open("data/words.txt", 'rb') as labels:
       for line in labels.read().splitlines()[18:]:
            match = re.findall(pattern,line.decode())[0]
            words.append(match)
    return words

def get_reshaped_images(X)
    X_reshaped = []
    for i,img in enumerate(X):
        try:
            img = cv2.resize(img,(200,200))
            X_reshaped.append(img)
        except:
            del Y[i]
    X_reshaped = np.array(X_reshaped)
    return X_reshaped

def get_vocab(words)
    word_to_index = {}
    index_to_word = {}
    i = 0
    for label in words:
        if label not in word_to_index:
            word_to_index[label] = i
            index_to_word[i] = label
            i += 1
    return word_to_index, index_to_word

def get_labels(words,word_to_index):
    Y = [word_to_index[word] for word in words]
    return Y

