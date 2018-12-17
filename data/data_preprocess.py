import numpy as np
import cv2
import sys
import glob
import re

def get_dataset(root, resize = 200, print_logs = True):
    """
    Returns X an array including all of the resized images and 'words' the associated string labels.
    """
    filenames = glob.glob(root+"/words/*/*/*.png")
    X = []
    pattern = "[^ ]+$"
    words = []
    total = len(filenames)
    increment = int(total/50)

    with open(root+"/words.txt", 'rb') as labels:
       for line in labels.read().splitlines()[18:]:
            match = re.findall(pattern,line.decode())[0]
            words.append(match)

    for i,file in enumerate(filenames):
        img = cv2.imread(file, 0)
        if img is not None:
            img = cv2.resize(img, (resize, resize))
            X.append(img)
        else:
            del words[i]
        if print_logs:
            sys.stdout.write("\r Data preprocessing : [" + "=" * int((i / increment)) + " " * int((((total - i) / increment))) + "] "
                             + str(round((i / total)*100,1)) + "%")
            sys.stdout.flush()

    X = np.array(X)
    return X, words

def get_vocab(words):
    word_to_index = {}
    index_to_word = {}
    i = 0
    for label in words:
        if label not in word_to_index:
            word_to_index[label] = i
            index_to_word[i] = label
            i += 1
    return word_to_index, index_to_word