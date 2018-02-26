#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# How many data points (people)?
print len(enron_data)

# For each person, how many features are available?
print len(enron_data["SKILLING JEFFREY K"])

# How many POIs are there in the E+F dataset?
print len(dict((key, value) for key, value in enron_data.items() if value["poi"] == True))

# How many POIs are there in total?
poi_reader = open('../final_project/poi_names.txt', 'r')
poi_reader.readline()
poi_reader.readline()

count = 0
for poi in poi_reader:
    count+=1
print count
#What is the total value of the stock belonging to James Prentice?
print enron_data["PRENTICE JAMES"]["total_stock_value"]
#How many email messages do we have from Wesley Colwell to persons of interest?
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
#Whats the value of stock options exercised by Jeffrey K Skilling?
print enron_data["SKILLING JEFFREY K"]['exercised_stock_options']
print ' {} , {}'.format(enron_data["SKILLING JEFFREY K"]['total_payments'] , 'skilling')
print ' {} , {}'.format(enron_data["LAY KENNETH L"]['total_payments'] , 'ken lay')
print ' {} , {}'.format(enron_data["FASTOW ANDREW S"]['total_payments'] , 'fastow')

#How many folks in this dataset have a quantified salary? What about a known email address?
print len(dict((key,value) for key, value in enron_data.items() if value["salary"] !='NaN'))
print len(dict((key,value) for key, value in enron_data.items() if value["email_address"] !='NaN'))
#What percentage of people in the dataset as a whole is this?
nan_salary = len(dict((key,value) for key, value in enron_data.items() if value["total_payments"] =='NaN'))
total = len(dict((key,value) for key, value in enron_data.items()))
percent = float(nan_salary) / total
print percent
#How many POIs in the E+F dataset have Nan for their total payments? What percentage of POIs as a whole is this?
pois = dict((key,value) for key, value in enron_data.items() if value["poi"] == True)
poi_payments = len(dict((key, value) for key, value in pois.items() if value["total_payments"] == 'NaN'))
percents = float(poi_payments)/len(pois)
print poi_payments
print percents

#What is the new number of people of the dataset? What is the new number of folks with Nan for total payments?
print len(enron_data) + 10
print 10 + len(dict((key, value) for key, value in enron_data.items() if value["total_payments"] == 'NaN'))

pois_num = len(pois) + 10
percent = float(10)/pois_num
print pois_num
print percent