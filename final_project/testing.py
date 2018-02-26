#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
from ggplot import *

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".




financial_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
features_list = financial_features + email_features
email_features.insert(0,'poi')
### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
#Sort Data into DataFrames
df = pd.DataFrame.from_records(list(data_dict.values()))
persons = pd.Series(list(data_dict.keys()))
print df.head()

# statistics about the data set
print 'Number of people:', df['poi'].count()
print 'Number of POIs:', df.loc[df.poi == True, 'poi'].count()
print 'Number of features:', df.shape[1]

#The counts of NaNs in each column
df = df.replace('NaN', np.nan)
print 'Missing data in each column \n',df.isnull().sum()
data_dict.pop('LOCKHART EUGENE E', 0)


data_points = len(data_dict)


### your code below

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

def removeMaxSalary():
    maxSalary = 1
    person_remove = None
    for person in data_dict:
        if data_dict[person]['salary'] != 'NaN' and data_dict[person]['salary'] > maxSalary:
            maxSalary  = data_dict[person]['salary']
            person_remove = person
    data_dict.pop(person_remove,0)
    print "Removed {} with salary {}".format(person_remove,maxSalary)

removeMaxSalary()
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

### Task 2: Remove outliers

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

### Task 4: Try a varity of classifiers
def scaleFeatures(arr):
    for i in range(1,arr.shape[1]):
        arrmin = min(arr[:,i])
        arrmax = max(arr[:,i])
        if arrmin == arrmax:
            arr[:,i] = arr[:,i]/arrmin
        else:
            arr[:,i] = (arr[:,i]-arrmin)/(arrmin-arrmax)
    return arr  


def get_best_features(data_dict, features_list):

    from feature_format import featureFormat
    from feature_format import targetFeatureSplit
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    
    my_dataset = data_dict
    
    data = featureFormat(my_dataset, features_list)
    
    data = scaleFeatures(data)
    
    labels, features = targetFeatureSplit(data)

    ## Below code from http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=500,
                                  random_state=0)
    
    forest.fit(features, labels)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(16):
        print("%d. feature %d (%f) -- %s" % (f + 1, indices[f], importances[indices[f]], features_list[f + 1]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(16), importances[indices][:16], align="center")
    plt.xticks(range(16), indices)
    plt.xlim([-1, 10])
    plt.show()
    ## end of "borrowed code" from scikit-learn
print get_best_features(data_dict,features_list)
   # Provided to give you a starting point. Try a varity of classifiers.
   
features_list = ["poi", "salary", "bonus", "expenses", "deferral_payments"]


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
def test_clf(clf,features_list,data_dict):
    data = featureFormat(data_dict, features_list)
    data = scaleFeatures(data)
    labels, features = targetFeatureSplit(data)
    from sklearn.cross_validation import StratifiedKFold
    skf = StratifiedKFold( labels, n_folds=3 )
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    for train_idx, test_idx in skf: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        
        print pred
        accuracies.append( accuracy_score(labels_test, pred) )
        precisions.append( precision_score(labels_test, pred) )
        recalls.append( recall_score(labels_test, pred) )
        f1s.append( f1_score(labels_test, pred) )
        print "accuracy score: ", accuracy_score( labels_test, pred )
        print "precision score: ", precision_score( labels_test, pred )
        print "recall score: ", recall_score( labels_test, pred)
        scores = {'accuracy': sum(accuracies)/3., 'precision': sum(precisions)/3.,
              'recall': sum(recalls)/3., 'f1': sum(f1s)/3. }
      
    ### aggregate precision and recall over all folds
        print "features_list = ", features_list
        print
        print "  * average accuracy: ", sum(accuracies)/3.
        print "  * average precision: ", sum(precisions)/3.
        print "  * average recall: ", sum(recalls)/3.
        print "  * average f1 score: ", sum(f1s)/3.
        #print "  * feature_importances_: ",sum(importances)/3.
        #return sum(importances)/10., scores
        return scores
      

   
def select_algorithm(data_dict, features_list, algo):

    from feature_format import featureFormat
    from feature_format import targetFeatureSplit
   
    ### store to my_dataset for easy export below
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list)

    # scale features
    data = scaleFeatures(data)
    
    ### split into labels and features (this line assumes that the first
    ### feature in the array is the label, which is why "poi" must always
    ### be first in features_list
    labels, features = targetFeatureSplit(data)
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    if algo == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif algo == 'RandomForest':
        clf = RandomForestClassifier()
    elif algo == 'DecisionTree':
        clf = DecisionTreeClassifier()
    return clf

algorithms = ['AdaBoost','RandomForest','DecisionTree']

for algo in algorithms:
    clf = select_algorithm(data_dict,features_list,algo)
    print clf
    print test_clf(clf,features_list,data_dict)
def algorithm_tuning(data_dict,features_list,iters=1000,rate=1.0):
    # scale features
   # parameters = {'num_iters': [5,10,25,50,100,150,200,250,350,500,750,1000,2000,5000],
              #    'learn_rate':[0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.6, 0.75,0.9,1,1.1,1.25,1.5,2,2.5,3]
              #    }
    ### split into labels and features (this line assumes that the first
    ### feature in the array is the label, which is why "poi" must always
    ### be first in features_list
    ### store to my_dataset for easy export below
    my_dataset = data_dict
    data = featureFormat(my_dataset, features_list)
    data = scaleFeatures(data)
    labels, features = targetFeatureSplit(data)
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier( random_state = 202, n_estimators = iters,learning_rate=rate)
    return clf
def tuning_iterations():
    num_iters = [5,10,25,50,100,150,200,250,350,500,750,1000,2000,5000]
    x = []
    y = []  
    f1 = []
    prec = []
    recall = []
    for n in num_iters:
        clf = algorithm_tuning(data_dict,features_list,n)
        print clf
        scores = test_clf(clf,features_list,data_dict)
        print scores
        y.append(scores['accuracy'])
        x.append(n)
        f1.append(scores['f1'])
        prec.append(scores['precision'])
        recall.append(scores['recall'])
    plt.plot(x,y)
    plt.show()
    plt.plot(x,f1)
    plt.show() 
    plt.plot(x,recall)
    plt.show()
    plt.plot(x,prec)
    plt.show()       
#tuning_iterations()
def tuning_rate():
    learn_rate = [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.6, 0.75,0.9,1,1.1,1.25,1.5,2,2.5,3]
    x = []
    y = []  
    f1 = []
    prec = []
    recall = []
    for r in learn_rate:
        clf = algorithm_tuning(data_dict,features_list,iters=1000,rate=r)
        print clf
        scores = test_clf(clf,features_list,data_dict)
        print scores
        y.append(scores['accuracy'])
        x.append(r)
        f1.append(scores['f1'])
        prec.append(scores['precision'])
        recall.append(scores['recall'])
    plt.plot(x,y)
    plt.show()
    plt.plot(x,f1)
    plt.show() 
    plt.plot(x,recall)
    plt.show()
    plt.plot(x,prec)
    plt.show()
##tuning_rate()
# best precision with 1.0
    #AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          #learning_rate=1.0, n_estimators=1000, random_state=34)
clf = algorithm_tuning(data_dict,features_list)
print clf
scores = test_clf(clf,features_list,data_dict)
print scores
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

#test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.


#dump_classifier_and_data(clf, my_dataset, features_list)