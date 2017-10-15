#! /usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
seed = 2
from xgboost import XGBClassifier
import pickle

print "Reading test data"
test  = pd.read_csv('../data/new_test.csv')

print "Reading predictive model"
with open('../data/model.pkl','rb') as f:
	model = pickle.load(f)


ids = test.id
test.drop(['id'], inplace=True, axis=1)
test.drop([ col for col in test.columns if col.startswith('ps_cont') ],axis=1, inplace=True)

print "encoding data"
with open('../data/OneHotEncoder.clf', 'rb') as f:
	encoders = pickle.load(f)

enc_test  = None
for feature,encoder in zip(test.columns,encoders) :
	encoded = encoder.transform(test[feature].values.reshape(-1,1))
	if enc_test is None :
		enc_test = encoded
	else :
		enc_test = np.concatenate((enc_test, encoded), axis=1)

del test

print "predicting test data"
pred = model.predict(enc_test)

del enc_test

print "saving to file"
with open('../data/predictions', 'wb') as f :
	f.write('id,target')
	for i,p in zip(ids,pred) :
		f.write('\n%d,%d'%(i,p))
