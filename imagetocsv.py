# importing libraries
import csv
import cv2
import numpy as np
import os
import glob

images = [file for file in glob.glob("*.jpg")]
for i in images:
	print(i)
	
	im=cv2.imread(i)
	#print(im)
	gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
	gray=cv2.resize(gray,(28,28))
	#print(gray)
	gray=np.matrix(gray)
	gray1=gray.flatten()
	gray1=np.array(gray1[0])
	with open('output.csv', 'a') as csvfile:
		
		writer = csv.writer(csvfile)
			
		writer.writerow(gray1[0])
	

