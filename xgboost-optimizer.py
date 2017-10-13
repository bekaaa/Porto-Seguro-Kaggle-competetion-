from xgboost import XGBClassifier

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
    'eta':0.3, #step size shrinkage used in update to prevents overfitting. default:0.3, range[0-1]

    #---------------------------------------------------------------------------------------------------
    #Learning Task Parameters
    #------------------------------------------------------------------------------
    'objective': 'reg:logistic', # reg:linear, reg:logistic, binary:logistic, multi:softmax
    'eval_metric' : 'error', # error for binary class., merror for multiclass classification, "map" Mean Average Precesion.
    'random_seed' : 0,
    'n_jobs' : 4
}
GCV_params = {
	'n_jobs' : 4,
	'pre_dispatch' : 4,
	'cv' : 2,
	'verbose' : 50,
	'scoring' : 'f1'
}

def max_delta_step() :
	minr = 0
	maxr = 8
	while True :
		cv_params = { 'max_delta_step': range(minr,maxr+1,2) }
		best_params = apply_GSearch(cv_params)
		i = best_params['max_delta_step']

		if i == minr :
			cv_params = { 'max_delta_step': [ minr, minr+1] }
			best_params = apply_GSearch(cv_params)
			i = best_params['max_delta_step']
			params['max_delta_step'] = i
			return

		elif i == maxr :
			minr = maxr
			maxr += 6

		else :
			cv_params = { 'max_delta_step': [i-1, i, i+1] }
			best_params = apply_GSearch(cv_params)
			i = best_params['max_delta_step']
			params['max_delta_step'] = i
			return
#---------------------------------------------------------------------------------
def optimize_param(param_name, step, start, _max ) :
	end = start + 2 * step * 4

	while True :
		cv_params = { param_name : range( start, end + step, step * 2 ) }
		best_params = apply_GSearch(cv_params)
		i = best_params[param_name]

		if i == start :
			cv_params = { param_name : [ i, i + step ] }
			best_params = apply_GSearch(cv_params)
			i = best_params[param_name]
			params[param_name] = i
			return

		elif i == _max :
			cv_params = { param_name : [ i - step, i ] }
			best_params = apply_GSearch(cv_params)
			i = best_params[param_name]
			params[param_name] = i
			return

		elif i == end :
			start = end
			end = start + 2 * step * 2

		else :
			cv_params = { param_name : [ i-step, i, i+step ] }
			best_params = apply_GSearch(cv_params)
			i = best_params[ param_name ]
			params[ param_name ] = i
			return








def imbalanced() :
	cv_params = { 'max_delta_step': range(0,10,2) }
	model = XGBClassifier(**params)

	gsearch = GridSearchCV(model, param_grid = cv_params, **GCV_params )
	gsearch.fit(enc_train, target)
	if gsearch.best_params_['max_delta_step'] == 9 :
		while True :
			i = gsearch.best_params_['max_delta_step']
			cv_params['max_delta_step'] = np.arange(i,i+3)
			gsearch = GridSearchCV(model, param_grid = cv_params, **GCV_params )
			gsearch.fit(enc_train, target)
			if


	print gsearch.best_params_,'\n' ,gsearch.best_score_

= {

}
cv_params2 = {
    'scale_pos_weight': range(1,10,2)
}
cv_params3 = {
    'min_child_weight': range(3,10,2),
    'max_depth' : range(3,10,2)
}
cv_params4 = {
    'min_child_weight': [4,3],
    'max_depth' : [10,11,12]
}
cv_params5 = {
    'gamma': np.arange(0, 0.5, 0.1)
}
cv_params6 = {
    'subsample': np.arange(0.1, 0.6, 0.1),
    'colsample_bylevel': np.arange(0.1, 0.6, 0.1)
}
cv_params7 = {
 'subsample':[.35,.4,.45],
 'colsample_bylevel':[.35,.4,.45]
}
cv_params8 = {
 'subsample':[.35,.4,.45],
 'colsample_bylevel':[.35,.4,.45]
}
cv_params9 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100],
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}







def get_model(parms):
    model = XGBClassifier(**default_parms)
