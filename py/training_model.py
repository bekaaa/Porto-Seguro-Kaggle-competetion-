#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
seed = 2
from xgboost_training import XGBClassifier
import pickle

print "reading data"
train = pd.read_csv('../data/new_train.csv')



#sample = train[train.target == 1]
#for ii in range(26) :
#	sample = sample.append( train[train.target == 1] )

#sample = sample.append( train[train.target == 0].sample(n=26 * 21694, random_state=seed) )
#sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
#train = sample

target = train.target
train.drop(['target','id'], inplace=True, axis=1)
train.drop([ col for col in train.columns if col.startswith('ps_cont') ],axis=1, inplace=True)

with open('../data/OneHotEncoder.clf', 'rb') as f:
	encoders = pickle.load(f)

print "encoding data"
enc_train = None
for feature,encoder in zip(train.columns,encoders) :
	encoded = encoder.transform(train[feature].values.reshape(-1,1))
	if enc_train is None :
		enc_train = encoded
	else :
		enc_train = np.concatenate((enc_train, encoded), axis=1)

del train

with open('../data/parameters.pkl','rb') as f:
	param = pickle.load(f)

param['eta'] = 0.01
model = XGBClassifier(**param)
print "fitting model"
model.fit(enc_train, target)

print "saving model to file"
with open('../data/model.pkl', 'wb') as f:
	pickle.dump(file=f, obj=model)

print "predicting train data"
pred = model.predict(enc_train)
from sklearn.metrics import classification_report
print classification_report(y_true=target, y_pred=pred)
