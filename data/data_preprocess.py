import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import h5py

def img_to_vect(img_path):
    img = cv2.imread(img_path,0)
    img_flatten = np.reshape(img,img.shape[0]*img.shape[1],1)
    return img_flatten

def get_max_shape():
    path = os.path.dirname(os.path.abspath("__file__")) + '/words'
    authors = os.listdir(path)
    max_width,max_height = 0,0
    for author in authors:
        path_author = path + '/' + author
        author_letters = os.listdir(path_author)
        for letter in author_letters:
            path_letter = path_author + '/' + letter
            words_img = os.listdir(path_letter)
            for img_path in words_img:
                img = cv2.imread(path_letter + '/' + img_path,0)
                try:
                    height,width = img.shape
                    if height > max_height:
                        max_height = height
                    if width > max_width:
                        max_width = width
                except:
                    continue
    return max_height, max_width

def create_dataset():
    path = os.path.dirname(os.path.realpath(__file__)) + '/words'
    nb_letters = 0
    authors = os.listdir(path)
    dataset = []
    for author in authors:
        path_author = path + '/' + author
        author_letters = os.listdir(path_author)
        nb_letters += len(author_letters)
        for letter in author_letters:
            path_letter = path_author + '/' + letter
            words_img = os.listdir(path_letter)
            try :
                text = [img_to_vect(path_letter + '/' + img_path) for img_path in words_img]
                dataset.append(text)
            except:
                print(path_letter)
    print("The dataset includes " + str(nb_letters) + " letters.")
    return dataset

# dataset = create_dataset()
#
# max_len_seq = 0
# max_len_img = 0
# m = len(dataset)
# for list in dataset:
#     if len(list) > max_len:
#         max_len = len(list)
#     for img in list:
#         if len(img)>max_len_img:
#             max_len_img = len(img)
#
# X = np.zeros(m,max_len,max_len_img)



