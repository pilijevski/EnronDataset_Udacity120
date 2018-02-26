#!/usr/bin/python

import sys
import pickle
import pandas
import numpy as np
sys.path.append("../tools/")
import sklearn
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
financial_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Load the POI Names file
poi_names = open("poi_names.txt", "r")

### Testing if it works
print data_dict["SKILLING JEFFREY K"]

#Sort Data into DataFrames
df = pandas.DataFrame.from_records(list(data_dict.values()))
persons = pandas.Series(list(data_dict.keys()))
print df.head()

# statistics about the data set
print 'Number of people:', df['poi'].count()
print 'Number of POIs:', df.loc[df.poi == True, 'poi'].count()
print 'Number of features:', df.shape[1]

#The counts of NaNs in each column
df = df.replace('NaN', np.nan)
print 'Missing data in each column \n',df.isnull().sum()
print "before delete",features_list
for column, series in df.iteritems():
    if series.isnull().sum() > 65:
        df.drop(column, axis=1, inplace=True)
        for item in features_list:
            if item == column:
                features_list.remove(item)
print "after delete",features_list
data_points = len(data_dict)
print 'Total data points(names) {}\t'.format(data_points)


######
# removing outliers
# ####

#function for plotting the outliers
def plot_outliers(data,a,b,a_name,b_name,pos):
    plt.subplot(4,4,pos)
    f1 = []
    f2 = []
    y = []
    for point in data:
        f1.append(point[a])
        f2.append(point[b])
        c = 'red' if point[0]==True else 'blue'
        y.append(c)
    plt.scatter(f1, f2, c=y)   
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    plt.xlabel(a_name)
    plt.ylabel(b_name)
## sporedba so salary i to_messages
def visualize_outliers():
    financial_outliers = featureFormat(data_dict, financial_features)
    email_outliers = featureFormat(data_dict, email_features)
    counter = 1
    for i in range(2, len(financial_features)):
        plot_outliers(financial_outliers, 1, i, 'salary', financial_features[i], counter)
        counter += 1
    plt.show()
    counter = 1
    for i in range(2, len(email_features)):
        plot_outliers(email_outliers, 1, i, 'to_messages', email_features[i], counter)
        counter += 1
    plt.show()

def removeMaxSalary():
    maxSalary = 1
    person_remove = None
    for person in data_dict:
        if data_dict[person]['salary'] != 'NaN' and data_dict[person]['salary'] > maxSalary:
            maxSalary  = data_dict[person]['salary']
            person_remove = person
    data_dict.pop(person_remove)
    print "Removed {} with salary {}".format(person_remove,maxSalary)

#plotting before removing outlier
#visualize_outliers()
#removing outlier from salary
removeMaxSalary()
#plotting after removing outlier
#visualize_outliers()

### Task 3: Create new feature(s)
def fraction_email(old,fraction):
    new_list=[]

    for data in data_dict:
        if data_dict[data][old]=="NaN" or data_dict[data][fraction]=="NaN":
            new_list.append(0.)
        elif data_dict[data][old]>=0:
            new_list.append(float(data_dict[data][old])/float(data_dict[data][fraction]))
    return new_list

### create two lists of new features
fraction_from_poi_email=fraction_email("from_poi_to_this_person","to_messages")
fraction_to_poi_email=fraction_email("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for data in data_dict:
    data_dict[data]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[data]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1
features_list.remove('from_this_person_to_poi')
features_list.remove('from_poi_to_this_person')
features_list.append('fraction_to_poi_email') 
features_list.append('fraction_from_poi_email')
features_list.remove('from_messages')
features_list.remove('to_messages')

print features_list
print len(features_list)

### Store to my_dataset for easy export below.
my_dataset = data_dict
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print 'accuracy', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"



importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(11):
    print "{} feature {} ({})".format(i+1,features_list[i+1],importances[indices[i]])
features_list.remove('other')
features_list.remove('expenses')
features_list.remove('restricted_stock')
# Provided to give you a starting point. Try a variety of classifiers.
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),10)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]
    
    # #decisionTree
     from sklearn.tree import DecisionTreeClassifier
     parameters = {'min_samples_split':[2,3,4,5,6,7,8,9]}
     cltree = DecisionTreeClassifier()
     clf = GridSearchCV(cltree, parameters)
     t0 = time()
     clf.fit(features,labels)
     print "training time:", round(time()-t0, 3), "s"
     t1 = time()
     pred = clf.predict(features)
     print "training time:", round(time()-t1, 3), "s"
     print  "DecisionTree accuracy",accuracy_score(pred,labels)
     print "DecisionTree best params:",clf.best_params_
     print "Precision score:",precision_score(labels_test,pred)
     print "Recall score",recall_score(labels_test,pred)


#Gaussian Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print accuracy
print "NB algorithm time:", round(time()-t0, 3), "s"
print "Precision score:",precision_score(labels_test,pred)
print "Recall score",recall_score(labels_test,pred)


#adaBoost
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
parameters = {
    'criterion': ('gini', 'entropy'),
    'min_samples_leaf':range(1, 50, 5),
    'max_depth': range(1, 10),
    'n_estimators': range(1,100,10)
}
# rfc = RandomForestClassifier()
# clf = GridSearchCV(rfc, parameters)
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# print  "Adabooost classifier",accuracy_score(pred,labels_test)
# print "adaboost best params:",clf.best_params_
# print "Precision score:",precision_score(labels_test,pred)
# print "Recall score",recall_score(labels_test,pred)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# clf = RandomForestClassifier(n_estimators= 6, criterion='entropy', max_depth= 9, min_samples_leaf= 1)
# clf.fit(features_train,labels_train)
# pred = clf.predict(features_test)
# print  "RandomForest classifier:",accuracy_score(pred,labels_test)
# print "Precision score:",precision_score(labels_test,pred)
# print "Recall score",recall_score(labels_test,pred)
best_params = {
	'n_estimators': 4, 
	'base_estimator__criterion': 'gini', 
	'base_estimator__max_depth': 3, 
	'base_estimator__min_samples_leaf': 11}

clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42)
clf.set_params(**best_params)

print  "RandomForest classifier:",accuracy_score(pred,labels_test)
print "Precision score:",precision_score(labels_test,pred)
print "Recall score",recall_score(labels_test,pred)
clf = RandomForestClassifier(n_estimators= 11, criterion='entropy', max_depth= 9, min_samples_leaf= 1)
clf.fit(features_train,labels_train)
my_dataset = data_dict
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
dump_classifier_and_data(clf, my_dataset, features_list)