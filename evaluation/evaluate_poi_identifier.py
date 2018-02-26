#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
features_train  , features_test, labels_train , labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)

from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.grid_search import GridSearchCV
#decisionTree

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
preds = clf.predict(features_test)
accuracy = accuracy_score(preds,labels_test)
print(accuracy)
num_pois = 0
for pred in preds:
	if pred == 1:
		num_pois += 1 

true_positives = 0
false_positives = 0
false_negatives = 0

for pred,actual in zip(preds,labels_test):
    if pred == actual and actual==1:
        true_positives+=1
    elif pred ==1 and actual ==0:
        false_positives+=1
    elif  pred==0 and actual==1:
        false_negatives+=1

print "Precision score:",precision_score(labels_test,preds)
print "Recall score",recall_score(labels_test,preds)

print "True Positives:", true_positives
print "False Positives", false_positives
print "False Negatives", false_negatives
print "Number of POIs predicted:", num_pois
print "Total number of people in test set:", len(preds)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
fn=0
tp=0
fp=0
for pred,actual in zip(predictions,true_labels):
    if pred == actual and actual==1:
        tp+=1
    elif pred ==1 and actual ==0:
        fp+=1
    elif  pred==0 and actual==1:
        fn+=1

print "\nPrecision (POIs):", tp*1.0/(tp + fp)
print "Recall (POIs)", tp*1.0/(tp + fn)