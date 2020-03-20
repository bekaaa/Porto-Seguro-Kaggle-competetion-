#!/usr/bin/env python
import numpy as np
from xgboost_training import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import log

train = None
target = None

params = {

	#-----------------------------------------------------------------------
	# dealing with imblanced data
	#-------------------------------------------------------------------------------
	'max_delta_step':0,
	#Maximum delta step we allow each tree's weight estimation to be. If the value is set to 0,
	#it means there is no constraint. If it is set to a positive value, it can help making the update step more
	#conservative. Usually this parameter is not needed, but it might help in logistic regression when class is
	#extremely imbalanced. Set it to value of 1-10 might help control the update.
	#default:0, range[0-inf]
	'scale_pos_weight' : 1,  # control balance between +ve and -ve weights, default:1

	#--------------------------------------------------------------------------------------
	#  Regularization
	#-------------------------------------------------------------------------------
	'alpha' : 0,
	#L1 regularization term on weights, increase this value will make model more conservative, default:0
	'lambda' : 1,
	#L2 regularization term on weights, increase this value will make model more conservative, default1.

	#------------------------------------------------------
	# add randomness to make training robust to noise
	#----------------------------------------------------------------
	'subsample' : 1,
	#ratio of the training instance. Setting it to 0.5 means that XGBoost randomly collected half of the data
	#instances to grow trees and this will prevent overfitting. default:1, range[0-1]
	'colsample_bytree' : 1, #subsample ratio of columns when constructing each tree.default:1, range[0-1]
	'colsample_bylevel' : 1, #subsample ratio of columns for each split, in each level. default:1, range[0-1]

	#-------------------------------------------------------------
	# Tree parameters
	#--------------------------------------------------
	'min_child_weight': 1, # minimum sum of instance weight (hessian) needed in a child.
	# the more conservative the algorithm will be. default:1, range[0-inf]
	'max_depth' : 6,
	# maximum depth of a tree, increase this value will make the model more complex / likely to be overfitting
	# default : 6, range[0-inf]

	#-----------------------------------------------------------------------
	# LOSS reduction
	#------------------------------------------
	'gamma':0, #minimum loss reduction required to make a further partition on a leaf node of the tree.
	#The larger, the more conservative the algorithm will be. default : 0, range[0-inf]
	'eta':0.01, #step size shrinkage used in update to prevents overfitting. default:0.3, range[0-1]

	#---------------------------------------------------------------------------------------------------
	#Learning Task Parameters
	#------------------------------------------------------------------------------
	'objective': 'reg:logistic', # reg:linear, reg:logistic, binary:logistic, multi:softmax
	'eval_metric' : 'error', # error for binary class., merror for multiclass classification, "map" Mean Average Precesion.
	'random_seed' : 0,
	'n_jobs' : 4
}

#---------------------------------------------------------------------------------
def optimize_param(param_name, start, step, _max = 20 ) :
	end = start + 2 * step * 4

	while True :
		cv_params = { param_name : np.arange( start, end + step, step * 2 ) }
		best_params, score = apply_GSearch(cv_params)
		i = best_params[param_name]

		if i == start :
			cv_params = { param_name : [ i, i + step ] }
			best_params, score = apply_GSearch(cv_params)
			i = best_params[param_name]
			params[param_name] = i
			return param_name, i, score

		elif i >= _max :
			cv_params = { param_name : [ i - step, i ] }
			best_params, score = apply_GSearch(cv_params)
			i = best_params[param_name]
			params[param_name] = i
			return param_name, i, score

		elif i == end :
			start = end
			end = start + 2 * step * 2

		else :
			cv_params = { param_name : [ i-step, i, i+step ] }
			best_params, score = apply_GSearch(cv_params)
			i = best_params[ param_name ]
			params[ param_name ] = i
			return param_name, i, score
#---------------------------------------------------------------------------------------
def apply_GSearch(cv_params) :

	model = XGBClassifier(**params)
	gsearch = GridSearchCV(model, param_grid = cv_params,
		cv = 2, verbose = 50, scoring='f1')
	gsearch.fit(train, target)
	return gsearch.best_params_, gsearch.best_score_
#----------------------------------------------------------------------------------------
def logmsg(n,v,s) :
		log.add( "best value for parameter : " + str(n) +
		" is " + str(v) + " , with score " + str(s) )
#--------------------------------------------------------------------------------------
def xgboost_optimizer(trainInput, targetInput):

	log.init('xgboost_optimizer.log')
	log.add('**************************************')
	log.add('log file initialized')
	#----------------------------------------------------------
	global train
	global target
	train = trainInput
	target = targetInput
	#---------------------------------------------------------------

	n,v,s = optimize_param('max_delta_step', 0, 1)
	logmsg(n,v,s)
	n,v,s = optimize_param('scale_pos_weight', 1, 1)
	logmsg(n,v,s)
	#------------------------------------------------

	n,v,s = optimize_param('min_child_weight', 0, 1)
	logmsg(n,v,s)

	n,v,s = optimize_param('max_depth', 0, 1)
	logmsg(n,v,s)
	#--------------------------------------------------

	n,v,s = optimize_param('gamma', 0, .1, 2)
	logmsg(n,v,s)
	#---------------------------------------------------

	n,v,s = optimize_param('subsample', 0, .05, 1)
	logmsg(n,v,s)

	n,v,s = optimize_param('colsample_bylevel', 0.1, .05, 1)
	logmsg(n,v,s)
	#----------------------------------------------------------

	n,v,s = optimize_param('reg_alpha',0, 0.1, 1)
	logmsg(n,v,s)

	n,v,s = optimize_param('reg_lambda',1, 1, 10)
	logmsg(n,v,s)

#----------------------------------------------------------------------------------------
	log.add("saving parameters to file")
	with open('../data/parameters.pkl', 'wb') as f:
		pickle.dump(file=f, obj=params)
	log.add("Done.")
	log.close()
#-------------------------------------------------------------------------------------
if __name__ == '__main__' :
	import pandas as pd
	from sklearn.preprocessing import OneHotEncoder
	seed = 0
	train = pd.read_csv('../data/new_train.csv')
	#sample_size = int(1e4)

	#sample = train[train.target == 1].sample(n=sample_size/2, random_state=seed)
	#sample = sample.append( train[train.target == 0].sample(n=sample_size/2, random_state=seed))
	#sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
	#train = sample

	target = train.target
	train.drop(['target','id'], inplace=True, axis=1)
	train.drop([ col for col in train.columns if col.startswith('ps_cont') ],axis=1, inplace=True)

	with open('../data/OneHotEncoder.clf', 'rb') as f:
		encoders = pickle.load(f)

	enc_train = None
	for feature,encoder in zip(train.columns,encoders) :
		encoded = encoder.transform(train[feature].values.reshape(-1,1))
		if enc_train is None :
			enc_train = encoded
		else :
			enc_train = np.concatenate((enc_train, encoded), axis=1)

	xgboost_optimizer(enc_train, target)
