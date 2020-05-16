# MNIST DATA

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import tensorflow as tf

#from keras.callbacks import ReduceLROnPlateau
    
# definig all the function
class mycallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self,epoch,log={}):
		if (log.get('acc')>.996):
			print('\nthe accuracy is greater than 99.6% so cancelling' )
			self.model.stop_training = True

callbacks=mycallback()
# DATA PREPROCESSING
    
# import data 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()                         # top 5 data of training dataset
print(train.shape)                   # shape of training data 
        
# cleaning the dataset 
target = train['label']                # storing the label column.
train = train.drop('label', axis=1)    # removing the label column from data 
target.value_counts()                  # counting number of same digits
        
# checking for any null or missing value
train.isnull().any().describe()        
test.isnull().any().describe()
        
# Normalizing the data 
train = train/255.0            # dividing by 255 bcoz the pixel highest value is 255
test = test/255.0
        
# reshaping the data 
train = train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
    
# spliting the data into train and test set
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(train, target, test_size = 0.1)
        
# tring to visuallize /plot the values of y_train
print(x_train[5])
plt.imshow(x_train[5][:,:,0], cmap = 'gray')  # done by me 
g = plt.imshow(x_train[0][:,:,0])
        
# CNN 
         
# defining the model using tensorflow
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128,activation = 'relu'),
	tf.keras.layers.Dense(10,activation = 'softmax')

])
# compiling the data to the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
# fit the training data to the model
model.fit(x_train,y_train,epochs=20, callbacks=[callbacks])
        
# to predict the model 
prediction = model.predict(x_val)

# confusion matrix and other parameter to analyze the accuracy of your model
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_val, prediction)
print(results)

