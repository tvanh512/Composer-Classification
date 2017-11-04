
# coding: utf-8

# In[ ]:

import music21,os,glob
from music21 import *
basedir = '/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/new_dataset'
ds = features.DataSet(classLabel='Class')
fes= [features.jSymbolic.PitchClassDistributionFeature,features.jSymbolic.ChangesOfMeterFeature,features.jSymbolic.InitialTimeSignatureFeature]
#fes = features.extractorsById(['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16','p19','p20','p21','r31','r32','r33','r34','r35'])
#fes = features.extractorsById(['q11','q12','q13'])
ds.addFeatureExtractors(fes)
i = 1
subs = ['byrd','chopin','pachelbel','tchaikovsky']
for sub in subs:
    for fname in glob.glob(os.path.join(basedir,sub,'*.xml')):
        print fname # fname is full path name
        #s = converter.parse(fname,classValue=0)
        s = converter.parse(fname)
        ds.addData(s,classValue=sub,id=str(i))
        i = i + 1
    ds.process()
    ds.write('/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/outputs/Byrd_Chopin_Pachelbel_Tchaikovsky.csv')


# In[1]:

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

filename = '/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/outputs/Byrd_Chopin_Pachelbel_Tchaikovsky.csv'
raw_data = open(filename, 'rb')
dataset = pandas.read_csv(filename)
#dataset = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
print type(dataset)
array = dataset.values
print type(array)
X = array[:,0:16]
Y = array[:,16]


# In[ ]:

import random
from random import shuffle
import numpy as np
from sklearn import cross_validation
# shuffle all train files
zipped = zip(X,Y)
random.shuffle(zipped)
X,Y = zip(*zipped)
X = np.asarray(X)
Y = np.asarray(Y)
print X.shape
print Y


# In[17]:

scoring = 'accuracy'
models = []
models.append(('Logistic Regression', LogisticRegression(C=1e5)))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
kf = cross_validation.KFold(len(Y), n_folds=10)
for name, model in models:
    print name
    c = [[0,0],[0,0]]
    acc = []
    cma = []
    cm = 0
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        acc.append(accuracy_score(Y_test, model.predict(X_test)))
        cma.append(confusion_matrix(Y_test, model.predict(X_test)))
        cm = cm + confusion_matrix(Y_test, model.predict(X_test))
        #b = classification_report(Y_test, model.predict(X_test))
        #print type(b)    
    acc = np.asarray(acc)
    print 'confusion matrix',cm
    print 'average accuracy', np.mean(acc)
    print 'average standard deviation', np.std(acc)

