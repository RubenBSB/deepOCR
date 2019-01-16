# hw-text-recognition

This is a school project in deep learning I am currently working on.

It consists in building a handwritten text recognition system using CNC-LSTM-CTC architecture.

This [article](https://arxiv.org/pdf/1411.4389.pdf) was really helpful to understand the concept of Convolutional Recurrent Neural Network (CRNN).

## Data

I am using the [IAM Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) which includes about 115,000 label images of English words from more than 1500 handwritten letters.

## Use

You have to register to download the dataset. Once it is done, unzip it and place the 'words' directory and 'words.txt' file in the project repository as following :

```
hw-text-recognition repository
├── data
│   ├── words
│   │   ├── a01
│   │   ├── a02
│   │   ├── ...
│   ├── words.txt
├── src
├── ...
```


