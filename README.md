# Handwritten-Digit-Recognition

Dataset is taken from kaggle here is the link https://www.kaggle.com/c/digit-recognizer/data

Dataset contain 2 file one is train.csv which is used to train our model and other is test.csv for testing purpose.
In this dataset image is converted into (1,784) pixel array for training purpose

when you visualize this array into (28,28) size 2D matrixs you will be able to see the image of number. 

For those who need to create there own dataset I have create Imagetocsv.py file which convert your new image into csv 
format which is required in this model. Simply put this file into you image folder and run this file, all the images 
will convert into (1,784) shape pixel array which then used directly to this model.

model.py file is the final file for training the model.

