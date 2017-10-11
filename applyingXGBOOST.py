#!/usr/bin/env python

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

seed = 40

train = pd.read_csv('./data/train.csv', na_values="-1")
test = pd.read_csv('./data/test.csv', na_values="-1")

#train = train.iloc[range(10000)]
train = train.sample(n=int(1e3), random_state=seed)
target = train.target
train.drop('target', inplace=True, axis=1)

testids = test.id

dropoutCols = ["ps_car_03_cat", "ps_car_05_cat"]
#train.drop(dropoutCols, inplace=True, axis=1)
#test.drop(dropoutCols, inplace=True, axis=1)

features = { 'ind' : [], 'car' :[], 'calc':[], 'reg':[] }
datatypes = { 'cat':[], 'bin':[], 'cont':[] }
import re
for i in train.columns :
	for j in features.keys() :
		if j in i : features[j].append(i)
	for j in datatypes.keys() :
		if j in i : datatypes[j].append(i)

	if re.match(re.compile('^ps_[a-z]+_[0-9]+$'), i) :
		datatypes['cont'].append(i)
#-------------------------------------
train = train[datatypes['bin']]
test = test[datatypes['bin']]


#x_train, x_cv, y_train, y_cv = train_test_split(train,target,test_size = .2, random_state=seed)

model = XGBClassifier(
	objective = 'binary:logistic',
	n_jobs = 4,
	random_state = seed,
	eval_metric = "error",
	#-------
	max_depth = 1,
	min_child_weight = 2,
	#----------
	gamma = 0,
	#----------
	subsample = 0.1,
	colsample_bytree = 0.1,
	#------------
	scale_pos_weight = 1,
	#-----------
	reg_alpha = 0,
	reg_lambda = 0,
	#------------
	learning_rate = 0.1,

	n_estimators = 1000,
	)

model.fit(train, target)
#with open('./data/model.pkl', 'rb') as f :
#	model = pickle.load(f)

#with open('./data/model.pkl', 'wb') as f :
#	pickle.dump(model, f)

print "finished fitting, starting the predicting phase."
pred = model.predict(test)
#print classification_report(y_cv, pred)

with open('./data/predictions', 'wb') as f :
	f.write('id,target')
	for i,l in zip( testids, pred ) :
		f.write('\n%d,%d'%(i,l))



GS_params1 ={
	'max_depth':range(3,10,2),
	'min_child_weight':range(3,10,2)
}
GS_params2 ={
	'max_depth':[6,7,8]
}

GS_params3 ={
	'gamma':[i/10.0 for i in range(0,5)]
}
GS_params4 = {
 'subsample':[i/10.0 for i in range(1,6)],
 'colsample_bytree':[i/10.0 for i in range(1,6)]
}
GS_params5 = {
 'subsample':[.55,.6,.65],
 'colsample_bytree':[.55,.6,.65]
}
GS_params6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
GS_params7 = {
'reg_alpha':[0, 1e-7,  1e-6],
 'reg_lambda':[0, 1e-7,  1e-6]
}
#gsearch = GridSearchCV(model, param_grid = GS_params7,n_jobs=4,pre_dispatch=4, cv=3, verbose=50 )
#gsearch.fit(train, target)
#print gsearch.grid_scores_, '\n' ,gsearch.best_params_,'\n' ,gsearch.best_score_
