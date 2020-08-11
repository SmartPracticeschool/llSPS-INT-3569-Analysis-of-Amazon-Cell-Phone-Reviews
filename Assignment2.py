# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk   
nltk.download('stopwords') 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
import numpy as np
import pickle
#import h5py  
#from tenpy.tools import hdf5_io

dataset = pd.read_csv('E:\\Intern_project_files\\20191226_reviews.csv', delimiter=',')
ps=PorterStemmer()
data=[]
for i in range(0,67986):
    review = dataset["body"][i]
    review = re.sub('[^a-zA-Z]', ' ', str(review))
    review=review.lower()
    review=review.split()
    #print(review)
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    data.append(review)


cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(data).toarray()
y=dataset.iloc[:,2].values
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=0)

model=Sequential()
model.add(Dense(kernel_initializer='uniform', activation='relu', input_dim=1500, units=20))
model.add(Dense(kernel_initializer='uniform', activation='sigmoid', units=1))
model.add(Dense(kernel_initializer='uniform', activation='relu',units=100))
model.add(Dense(kernel_initializer='uniform', activation='sigmoid',units=1))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=50,batch_size=32)
model.save('aishwaryaModel.h5')


model= load_model("C:\\Users\\AISHWARYA\\Desktop\\InternshipProejectFiles\\aishwaryaModel.h5")
#with open('aishwaryaModel.h5','rb') as file:
    #cv=pickle.load(file)
#with h5py.File('aishwaryaModel.h5','r') as file:
    #cv=hdf5_io.load_from_hdf5(file) 
#entered_input= request.GET['review']
entered_input= "the food is best"
x_intent=cv.transform([entered_input])
y_pred=model.predict(x_intent)
if(y_pred>0.5):
    print("it is a positive review")
else:
    print("it is a negative review")





