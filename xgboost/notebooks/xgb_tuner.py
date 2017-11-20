#!/usr/bin/env python
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import log


class xgb_tuner :
	#*******************************************************************************
	def __init__(self, dtrain, dvalid, params, logging = False, log_file_index = -1,
		rounds=400, esrounds=50, seed=0, nfolds=3) :
		#-------------------
		self.__dtrain = dtrain
		self.__dvalid = dvalid
		self.__logging = logging
		self.__rounds = rounds
		self.__esrounds = esrounds
		self.__nfolds = nfolds
		self.__seed = seed
		self.params = params
		np.random.seed(self.__seed)
		#-----------------
		self.best_score = -1
		self.cvfolds = None
		#--------------------
		if self.__logging : self.__init_log(log_file_index)
	#*****************************************************************************
	def __call__(self, param_names, param_grid) :
		'''
		call function whenver a user wants to start a grid-call search.
		it makes some assertions then passes the inputer to master function > __tune
		return : best grid choice.
		'''
		# some assertion to make sure of input.
		assert type(param_names) == list
		assert type(param_grid) == list
		assert len(param_names) == len(param_grid[0])
		for p in param_names : assert p in self.params.keys()
		#---------------------------------------
		self.best_grid = self.__tune(param_names, param_grid)
		#----------------------------------------
		return self.best_grid
	#*****************************************************************************
	def __tune(self, param_names, param_grid) :
		'''
		Master function, takes "parameter names" and grids as input,
		log and analyze them. Then for each grid run a self.step_cv()
		checks ouput for best score.
		return : best grid.
		'''
		# ------- emptying some variables ----------------
		best_sc = -1
		self.best_score = -1
		self.cvfolds = None
		#----------------------------------------
		if self.__logging : log.msg('**** Starting grid-call search *********')

		for grid in param_grid :
			#------ set starter log message -----------------
			msg = "CV with "
			for i,v in enumerate(grid):
				msg+= "%s = %g, " % (param_names[i], v)
			if self.__logging : log.msg(msg)
			#------------------------------------------------------------
			#-------- call step_cv function on each grid -----------------
			f_train_sc, f_dev_sc, dev_sc, n_trees, step_time = self.__step_cv(param_names, grid)
			#------------------------------------------------------------
			# ------------ another log message ---------------
			msg = 'fold-train score : %g, fold-dev score : %g, dev score : %g, n_trees : %d,'\
					'step time : %.1f minutes'\
					% (f_train_sc, f_dev_sc, dev_sc, n_trees, step_time)
			if self.__logging : log.msg(msg)
			else : print msg
			# -------------------------------------------------------
			# ---- checking for best score and assigning some other bests ;) ---------
			if dev_sc > best_sc :
				best_sc = dev_sc
				best_f_train = f_train_sc
				best_f_dev = f_dev_sc
				best_n_trees = n_trees
				best_grid = grid
			#-------------------------------------------------------------------------

		#-------- last log message for grid-call search ----------------------
		if self.__logging : log.msg('****** End of grid-call search *********')
		msg =   'best dev score : %g, '\
				'best fold-train score : %g, '\
				'best fold-dev score : %g, '\
				'best n_trees = %d, '\
			   % ( best_sc, best_f_train, best_f_dev, best_n_trees )
		for i,v in enumerate(best_grid) :
			msg += 'best %s = %g, ' % (param_names[i], v)
		msg += '\n-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n'
		if self.__logging : log.msg(msg)
		print msg
		#---------------------------------------------------------------------------
		return best_grid
	#***********************************************************************************
	def __step_cv(self, param_names, grid ) :
		''' it calls the xgb.cv function,
			input : parameter names and a list of values as the same size of names
			returns a set of : fold-train-score, fold-test-score, dev-score, best-n-trees,
			and time in minutes.
			~ analyses only on grid a time.
		 '''
		# update the params dict with the new grid.
		for p,v in zip(param_names, grid) :
			self.params[p] = v

		t = np.datetime64('now')
		self.cvfolds = None
		self.best_score = -1

		cv_results = xgb.cv(self.params, self.__dtrain,
						num_boost_round=self.__rounds,
						early_stopping_rounds=self.__esrounds,
						seed=self.__seed,
						nfold=self.__nfolds,
						stratified = True,
						metrics=('auc'),
						fpreproc=self.__fpreproc,
						verbose_eval = False,
						callbacks=[self.__GetBestCVFolds()]
						)
		#-----------------------
		# predict dev-set and get it's gini score
		assert self.cvfolds != None
		dev_preds = np.zeros(self.__dvalid.num_row())
		for fold in self.cvfolds :
			dev_preds += fold.bst.predict(self.__dvalid)
		dev_preds /= len(self.cvfolds)
		dev_score = self.gini(self.__dvalid.get_label(), dev_preds)
		#--------------------------------------
		return (np.max(cv_results['train-auc-mean']), # fold-train score
				np.max(cv_results['test-auc-mean']), # fold-dev score
				dev_score, # dev score
				np.argmax(cv_results['test-auc-mean']), # best number of trees
				(np.datetime64('now') - t).astype('int') / 60. # iteration time
				)
	#********************************************************************************
	def gini(self, labels, preds) :
		'''defining Gini's Score function.'''
		return roc_auc_score(labels, preds) * 2. - 1
	#**********************************************************************************
	def __init_log(self, index) :
		assert index >= 0
		if index == 0 : index = "test"
		log.LOG_PATH = './logs/'
		try :
			_ = log.close()
		except :
			pass
		log.init('tuning_params-' + str(index) + '.log')
		log.msg('------------------initialized-----------------')
	#************************************************************************************
	def __fpreproc(self, dtrain_, dtest_, param_):
		''' passes to xgb.cv as a preprocessing function '''
		label = dtrain_.get_label()
		ratio = float(np.sum(label == 0)) / np.sum(label == 1)
		param_['scale_pos_weight'] = ratio
		return (dtrain_, dtest_, param_)
	#*************************************************************************************
	def __GetBestCVFolds(self) :
		'''passes to xgb.cv as a callback'''
		#def init(env) :
		#	self.best_score = -1
		def callback(env) :
			current_score = env.evaluation_result_list[1][1]
			if current_score > self.best_score :
				self.best_score = current_score
				self.cvfolds = env.cvfolds
		callback.before_iteration = False
		return callback
	#****************************************************************************************
