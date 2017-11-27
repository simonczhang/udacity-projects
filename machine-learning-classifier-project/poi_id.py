#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



### features_list is a list of strings, each of which is a feature name.
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments',
                 'total_payments', 'exercised_stock_options', 'bonus',
                 'restricted_stock', 'shared_receipt_with_poi',
                 'restricted_stock_deferred', 'total_stock_value',
                 'expenses', 'loan_advances', 'from_messages',
                 'other', 'director_fees', 'from_poi_to_this_person',
                 'deferred_income', 'long_term_incentive',
                 'from_this_person_to_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###Answer the questions in free response section
if False:
    data_points = len(data_dict)
    
    count_poi = 0
    count_nonpoi = 0
    for i, v in data_dict.iteritems():
        if v['poi'] == True:
            count_poi +=1
        else:
            count_nonpoi += 1
    
    features_used = len(features_list)
    
    nan_dict = {}
    for i in features_list:
        count = 0
        for j in data_dict:
            if data_dict[j][i] == 'NaN':
                count += 1
                key = i 
                value = count
                nan_dict[key] = value
    
    print 'total number of data points:', data_points
    print 'Number of pois:', count_poi, 'Number of Non-pois:', count_nonpoi
    print 'number of features used:', features_used
    print 'features with significant missing values:\n', nan_dict


### Remove outliers

data_dict.pop('TOTAL', 0)


x_value = 'salary'
y_value =  'bonus'
plot_features = [x_value, y_value]

if False:     #just investigating outliers
    for i in data_dict:
        if data_dict[i]['salary'] != 'NaN' and data_dict[i]['bonus'] != 'NaN':
            if data_dict[i]['salary'] > 1000000 and data_dict[i]['bonus'] > 5000000:
                print i
                
if False:  #checking features for graph analysis for outliers
    for i in data_dict:
        if data_dict[i][y_value] != 'NaN':
            if data_dict[i][y_value] > 4000000:
                person = i
                number = data_dict[person][y_value]
                print y_value
                print 'test outlier:', person, number
                print 'poi?', data_dict[person]['poi']



data = featureFormat(data_dict, plot_features)
#print data
if False: #plotting scatterplot
    for point in data:
        x_axis = point[0]
        y_axis = point[1]
        matplotlib.pyplot.scatter( x_axis, y_axis )
    
    matplotlib.pyplot.xlabel(x_value)
    matplotlib.pyplot.ylabel(y_value)
    matplotlib.pyplot.show()


### Create new feature(s)
def compute_fraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """

    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0
    else:
        return float(poi_messages) / all_messages


for name in data_dict:

    data_point = data_dict[name]

    

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction( from_poi_to_this_person, to_messages )
    #print fraction_from_poi
    data_point["fraction_from_poi"] = fraction_from_poi



    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction( from_this_person_to_poi, from_messages )
    #print fraction_to_poi
    data_point["fraction_to_poi"] = fraction_to_poi
    

    
### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Try a varity of classifiers

from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score, precision_score, f1_score


'''
Tester.py results
BASE PARAMETERS TEST ONLY:
SVC:
    sucks got 0 True positives for base params

Adaboost:
    Precision: 0.39974     
    Recall: 0.30200
    F1: 0.34406
    Accuracy: 0.84647
    
NB:
    Precision: 0.23578
    Recall: 0.40000
    F1: 0.29668
    Accuracy: 0.74713
    
kNN:
    Precision: 0.65278
    Recall: 0.21150
    F1: 0.31949
    Accuracy: 0.87987
    
RandomForest:
    Precision: 0.34720     
    Recall: 0.10850
    F1: 0.16533
    Accuracy: 0.85393
    
    
    
    
    
BASE PARAMETERS TEST W/PCA(n_estimators=10) PIPELINE:
SVC:
    sucks still no results....
    
Adaboost:
    Precision: 0.19007 
    Recall: 0.13400
    F1: 0.15718   
    Accuracy: 0.80840   
    
NB:
    Precision: 0.42621     
    Recall: 0.30900
    F1: 0.35826 
    Accuracy: 0.85240
    
kNN:
    Precision: 0.67626     
    Recall: 0.23500 
    F1: 0.34879 
    Accuracy: 0.88300 
    
RandomForest:
    Precision: 0.35325    
    Recall: 0.13600
    F1: 0.19639 
    Accuracy: 0.85160
    

After testing the classifiers with just base parameters, and a pipeline with
pca and classifier with base parameters, I found that NB with PCA, and Adaboost
with base parameters were the only two that had precision and recall already
over 0.3. Will fine tune those two to see how much better I can make them. 
    
'''


###Pipe it up!!! Make pipeline
if True: #gaussianNB pipe
    estimators = [('scaler', MinMaxScaler()),
                  ('pca', PCA(n_components=9)), 
                  ('clf', GaussianNB())]
    
    pipe = Pipeline(estimators)


if False: #adaboost pipe
    estimators = [('feature_selection', SelectKBest()), 
                  ('clf', AdaBoostClassifier())]
    
    pipe = Pipeline(estimators)



#GaussianNB para grid. 
nb_param_grid = {
    'pca__n_components': (7, 9, 10, 15),
}



#Adaboost param grid
ada_param_grid = {
    'feature_selection__k': (5, 10, 15),
    'clf__n_estimators': (25, 50 , 75, 100),
}


#scoring list
scoring = ['f1', 'recall', 'precision']


### Tune classifier to achieve better than .3 precision and recall 
### using tester script.
for score in scoring:
    #make gridSearch        
    grid = GridSearchCV(pipe,
                       nb_param_grid,
                       scoring=score,
                       error_score=0, 
                       cv=5, 
                       refit=score)
    
    
    t0 = time()
    grid.fit(features, labels)
    print "time to fit/train:", round(time()-t0, 3), "s"
    
    
    
    print 'best tuning for', score, ':', grid.best_params_
    print 'best score for', score, ':', grid.best_score_
    clf = grid.best_estimator_



'''
Results for tester.py after parameter tuning

PCA, GaussianNB results:
Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', GaussianNB(priors=None))])
        Accuracy: 0.82140       Precision: 0.35584      Recall: 0.41900 F1: 0.38485     F2: 0.40464
        Total predictions: 15000        True positives:  838    False positives: 1517   False negatives: 1162   True negatives: 11483



PCA, GaussianNB with 'fraction_from_poi' and 'fraction_to_poi' features:
Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', PCA(copy=True, iterated_power='auto', n_components=9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)), ('clf', GaussianNB(priors=None))])
        Accuracy: 0.82013       Precision: 0.32374      Recall: 0.32050 F1: 0.32211     F2: 0.32114
        Total predictions: 15000        True positives:  641    False positives: 1339   False negatives: 1359   True negatives: 11661





SelectKBest, Adaboost results:
Pipeline(memory=None,
     steps=[('feature_selection', SelectKBest(k=10, score_func=<function f_classif at 0x1a104e0a28>)), ('clf', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=25, random_state=None))])
        Accuracy: 0.83913       Precision: 0.35365      Recall: 0.24950 F1: 0.29258     F2: 0.26512
        Total predictions: 15000        True positives:  499    False positives:  912   False negatives: 1501   True negatives: 12088
        
        



    
GaussianNB turned out better after tuning from base parameters. Adaboost seems to do much worse after tuning but probably because 
of the dataset. I tried MinMaxScaler on adaboost pipeline as well as selectPercentile. all results were pretty much worse than the base parameters...
Fully tuned GaussianNB still has the best results of all classifiers I tested. 
    

'''


### Dump classifier, dataset, and features_list so anyone can
### check results. 

dump_classifier_and_data(clf, my_dataset, features_list)