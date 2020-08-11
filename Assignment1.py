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
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('E:\\Intern_project_files\\20191226_reviews.csv', delimiter=',')
ps=PorterStemmer()
data=[]
for i in range(0,67986):
    review = dataset["body"][i]
    review = re.sub('[^a-zA-Z]', ' ', str(review))
    review=review.lower()
    review=review.split()
    print(review)
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    data.append(review)


cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(data).toarray()
y=dataset.iloc[:,2].values
X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.20,random_state=0)




