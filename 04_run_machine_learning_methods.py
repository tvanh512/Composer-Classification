import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config
from sklearn.model_selection import validation_curve
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from random import shuffle

array = np.load(os.path.join(outdir,'features_4_composers_final.npz'))
X1 = array['melody']
X2 = array['bass']
X3 = array['chord']
X4 = array['duration']

X= np.concatenate((X1,X2,X3,X4),axis = 1)
print X1.shape
print type(X1)
print X2.shape
print X3.shape
print X4.shape
print X.shape
import random

#Shuffle the data
shuffle_seed = 10
random.seed(shuffle_seed)
# shuffle all train files
zipped = zip(X,Y)
random.shuffle(zipped)
X,Y = zip(*zipped)
X = np.asarray(X)
Y = np.asarray(Y)

# Plot bass features
c = []
for composer in Y:
    if composer == 'byrd':
        c.append('red')
    if composer == 'chopin':
        c.append('blue')
    if composer == 'pachelbel':
        c.append('green')
    if composer == 'tchaikovsky':
        c.append('black')
plt.figure(figsize=(10,10))
#plt.axis([-0.002, 0.010, -0.005, 0.03])
plt.scatter(X[:,28276], X[:,33931],color =c)
classes = ['Byrd','Chopin','Pachelbel','Tchaikovsky']
class_colours = ['red','blue','green','black']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
plt.legend(recs,classes,loc=4)
plt.xlabel('Normalization count of tuple [4,4,4,4] in bass notes')
plt.ylabel('Normalization count of tuple [7,7,7,7] in bass notes')
plt.savefig("/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/outputs/graph_4_7.png",dpi = 400)
plt.show()

# Training and testing with proposed_features
scoring = 'accuracy'
models = []
models.append(('Logistic Regression', LogisticRegression(C=1e4,max_iter=500)))
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
    acc = np.asarray(acc)
    print 'confusion matrix',cm
    print 'average accuracy', np.mean(acc)
    print 'average standard deviation', np.std(acc)

# plot the curse show the relation between C and accuracy

param_range = [0.1,1,10,100,1000,10000,100000]
train_scores, test_scores = validation_curve(
    LogisticRegression(max_iter=500), X, Y, param_name="C", param_range=param_range,
    cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Logistic Regression performance")
plt.xlabel("Inverse of regularization strength")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training accuracy",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Testing accuracy",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig("/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/outputs/Validation_curve.png",dpi = 400)
plt.show()

# PLot the learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing accuracy")

    plt.legend(loc="best")
    return plt

title = "Logistic Regression Learning Curves "

estimator = LogisticRegression(C=1e4,max_iter=500)
plot_learning_curve(estimator, title, X, Y, (0.2, 1.01),cv=10, n_jobs=1)
plt.savefig("/Users/VietAnh/Documents/Courses/Machinelearning/Midterm/outputs/Learning_curve.png",dpi = 400)
plt.show()