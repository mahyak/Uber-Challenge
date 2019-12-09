#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:07:38 2019

@author: mahya
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from warnings import simplefilter
from sklearn.metrics import accuracy_score, precision_score,recall_score
simplefilter(action='ignore', category=FutureWarning)

def percent_missing(dataframe):
    missing = (dataframe.isnull().sum()/dataframe.shape[0])*100
    my_colors = ['g', 'b']*11
    my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*11
    my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(missing))]
    ax = missing.plot(kind='bar', title='missing value percentage', 
                 figsize=(10,8), fontsize=12, rot=70,color=my_colors)
    ax.set_xlabel('Features name', fontsize=12)
    ax.set_ylabel('Percentage of missing value', fontsize=12)
    plt.show()
    
def accuracy_calc (y_test, y_pred, classifier):
    print('Accuracy of {0} Calssification:'.format(classifier) )
    print('Accuracy: ', round(accuracy_score(y_test,predictions),2))
    print('Precission: ', round(precision_score(y_test,predictions),2))
    print('Recall: ', round(recall_score(y_test,predictions),2))
    print(classification_report(y_test,predictions))
    print('######################################################')

       

signups = pd.read_csv('uber_challenge.csv')
#print(signups.describe(include='all'))
#print(signups.shape)

percent_missing(signups)
#bgc = [pd.notnull(i)*1 for i in signups.bgc_date]
#vehicle_added = [pd.notnull(i)*1 for i in signups.vehicle_added_date]
#vehicle_year = [int(i) for i in signups.vehicle_year.fillna(0)]
#subdata = pd.DataFrame({'bgc':bgc,'vehicle_added':vehicle_added, 'vehicle_year':vehicle_year})
#
#has_driven = [pd.notnull(i)*1 for i in signups.first_completed_date]
#cities = pd.get_dummies(signups.city_name.fillna('UnknownCity'))
#os = pd.get_dummies(signups.signup_os.fillna('UnknownOS'))
#channel = pd.get_dummies(signups.signup_channel.fillna('UnknownChannel'))
#vehicle_make = pd.get_dummies(signups.vehicle_make.fillna('UnknownMake'))
#vehicle_model = pd.get_dummies(signups.vehicle_model.fillna('UnknownModel'))
#clean_data=pd.concat([cities,os,channel,vehicle_make,vehicle_model, subdata],axis=1)
#print ('Answer = ', round(sum(has_driven)/float(len(signups['id'])),2))
#

signups['first_drive'] = [pd.notnull(i)*1 for i in signups.first_completed_date]
signups_date = pd.to_datetime(signups['signup_date'], format = '%m/%d/%y')
bgc_date = pd.to_datetime(signups['bgc_date'], format = '%m/%d/%y')
signups['bgc_submitted_date'] = (bgc_date - signups_date).dt.days
signups = signups.dropna(subset = ['bgc_submitted_date'])
signups = signups.dropna(subset = ['signup_os'])
signups_clean = signups[['signup_date', 'signup_os', 'signup_channel',
                          'city_name', 'bgc_submitted_date', 'first_drive']]
print(signups_clean)
print(signups_clean['first_drive'].value_counts(normalize=True).mul(100).astype(str)+'%')

# =============================================================================
# signup channel produced first drivers per city
# =============================================================================
Berton = signups_clean.loc[signups_clean['city_name'] == 'Berton']
Strark = signups_clean.loc[signups_clean['city_name'] == 'Strark']
Wrouver = signups_clean.loc[signups_clean['city_name'] == 'Wrouver']


fig = plt.figure(figsize=(10,4))
title = fig.suptitle("signup channel produced first drivers per city", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.5)

x, hue = "signup_channel", "first_drive"
hue_order = ["True", "False"]
palette=["#f2c0ff", "#6915cf"]

ax1 = fig.add_subplot(1,3,1)
sns.countplot(x=x, hue=hue, data=Berton, ax = ax1, palette=palette)

ax2 = fig.add_subplot(1,3,2)
sns.countplot(x=x, hue= hue, data=Strark, ax = ax2,palette=palette)

ax3 = fig.add_subplot(1,3,3)
sns.countplot(x=x, hue= hue, data=Wrouver, ax = ax3,palette=palette)


# =============================================================================
# distribution of the timeto submit their background check consent form
# =============================================================================
distibution_tbgc = signups_clean['bgc_submitted_date'].loc[signups_clean['first_drive'] == True]
plt.hist(distibution_tbgc, facecolor='blue', alpha=0.7, bins='auto', rwidth=0.8)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('bgc_submitted_date')
plt.ylabel('Count')
plt.title('Distibution of Time Taken to Submit Background Check')
plt.show()

# =============================================================================
# most signup in time of the month
# =============================================================================
distibution_signup_time = signups_clean.loc[signups_clean['first_drive']==True]
plt.scatter(distibution_signup_time['signup_date'], distibution_signup_time['bgc_submitted_date'],
            color="blue",alpha=0.7)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('signup_date')
plt.ylabel('bgc_submitted_date')
plt.title('Distibution of Signup dates')
plt.show()

# =============================================================================
#  Chi-Squared Test
# =============================================================================

first_os = pd.crosstab(signups_clean['signup_os'], signups_clean['first_drive'])
print(first_os)
print(stats.chi2_contingency(first_os))

first_city = pd.crosstab(signups_clean['city_name'], signups_clean['first_drive'])
print(stats.chi2_contingency(first_city))

first_channel = pd.crosstab(signups_clean['signup_channel'], signups_clean['first_drive'])
print(stats.chi2_contingency(first_channel))

# =============================================================================
# Predicting Model
# =============================================================================
signups_clean = signups_clean.dropna()
signup_os = pd.get_dummies(signups_clean['signup_os'],drop_first=True)
signup_channel = pd.get_dummies(signups_clean['signup_channel'],drop_first=True)
signups_clean.drop(['signup_date','signup_os','signup_channel', 'city_name'], axis=1,inplace=True)
signups = pd.concat([signups_clean,signup_os,signup_channel],axis=1)
X_train, X_test, y_train, y_test = train_test_split(signups.drop('first_drive', axis=1), signups['first_drive'], 
                                                    test_size=0.33,random_state=43)

#X_train, X_test, y_train, y_test = train_test_split(clean_data, has_driven, test_size=0.33,random_state=43)


################  Logistic Regression   ################
clf = LogisticRegression(random_state=0)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
accuracy_calc(y_test, predictions, 'Logistic Regression') 

################  Decision Tree Classifier   ################
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
#tree.plot_tree(clf.fit(X_train, y_train))
accuracy_calc(y_test, predictions, 'Decision Tree')
 
################  AdaBoost Classifier   ################
svc = SVC(kernel = 'linear')
clf = AdaBoostClassifier(n_estimators=50, learning_rate=1)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy_calc(y_test, predictions, 'AdaBoost')

################  K-Nearest Neighbors Classifier   ################
clf = KNeighborsClassifier(n_neighbors=50)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy_calc(y_test, predictions, 'K-Nearest Neighbors')








