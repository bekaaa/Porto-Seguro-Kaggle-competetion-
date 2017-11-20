#!/usr/bin/env python
import numpy as np
import xgboost as xgb
from xgb_tuner import xgb_tuner
from sklearn.model_selection import train_test_split
import log
'''
Using the xgb_tuner Class. I will tune the important parameters in xgboost.
This should work fine for most of the problems.
You could change the values if you want.
I choosed values which makes sense, and tried my best to cover all possibilities.

Tuning goes on 4 stages :
1 - Handling Imbalanced data.
	** max_delta_step and scale_pos_weight
	** scale_pos_weight is already assigned for each fold in xgb_tuner() with the ratio of
		negative examples to the positive examples.
	** max_delta_step not important to tune, but it should be a value from 0 to 10.
	** see this https://github.com/dmlc/xgboost/blob/master/doc/how_to/param_tuning.md#handle-imbalanced-dataset

2 - Controling model complexity :
	** max_depth, min_child_weight, and gamma
	** and those i will tune them for a large scale of values.

3 - control model's robusting to noise.
	** subsample and colsample_bytree

4 - Regularization terms.
	** alpha and lambda

5 - learning rate and number of rounds.
	** eta and num_rounds
'''
class parametizer :
	def __init__(self, train_file, preproc = None, test_file=None, dev_size=3000,
		log_file_index=-1) :

		self.seed = 0
		#------- prepare log file -----------------#
		assert log_file_index >= 0
		self.init_log(log_file_index)
		log.msg('****************************************')
		log.msg('*** log file initialized ********')
		#----------------------------------------------
		# ------- preparing data ----------------------- #
		log.msg('* preparing data')
		try :
			self.train = np.load(train_file)
			self.test = np.load(test_file) if test_file else None
		except :
			raise ValueError('Wrong train/test file input')
		#self.labels = self.train[:,0]
		#self.train = self.train[:,1:]
		if preproc : self.train, self.labels, self.test = preproc(self.train, self.test)

		self.x_train, self.x_dev, self.y_train, self.y_dev = \
			train_test_split(self.train,self.labels,random_state=self.seed,test_size=dev_size)

		self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
		self.dvalid = xgb.DMatrix(self.x_dev, label=self.y_dev)
		self.dtest  = xgb.DMatrix(self.test) if self.test else None

		del self.train, self.labels, self.x_train, self.x_dev, self.y_train, self.y_dev, self.test
		log.msg('data is ready to use.')
		# ------------ data is ready ----------------- #
		# -------------------------------------------- #
		# ------ initializa the parameters ----------- #
		self.params = {
			'max_delta_step' : 0,
			'scale_pos_weight' : 1, # calculated for each fold. #neg / #pos

			'max_depth' : 6,
			'min_child_weight' : 1,
			'gamma' : 0,

			'subsample' : 1,
			'colsample_bytree' : 1,

			'reg_alpha' : 0,
			'reg_lambda' : 1,

			'eta' : 0.3,

			'objective' : "binary:logistic",
			'eval_metric' : 'auc',
			'n_jobs' : -1,
			'random_seed' : self.seed
		}
		self.rounds = 400
		self.esrounds = 50 # early stop rounds.
		# ------------------------------------------ #
		# ---- initializing xgb_tuner object ------ #
		self.tuner = xgb_tuner(self.dtrain, self.dvalid, self.params,
			logging=True, log_file_index=log_file_index,
			rounds=self.rounds, esrounds=self.esrounds)
		#----------------------------------------------------
		del self.dtrain, self.dvalid
		log.msg('class is ready.')
	##############################################################
	def doall(self) :
		'''
		tune all parameters.
		'''
		self.tune_data_imbalancing()
		self.tune_model_complexity()
		self.tune_model_robustness()
		self.tune_regulr_terms()
		self.tune_eta_rounds()
	##################################################################
	def init_log(self, index) :
		if index == 0 : index = "test"
		log.LOG_PATH = './logs/'
		try :
			_ = log.close()
		except :
			pass
		log.init('tuning_params-' + str(index) + '.log')
	########################################################################
	#######################################################################
	def tune_gamma(self, levels=3, init_step=1, start=0) :
		terms = ['gamma']
		#----------------------------------
		# example : 1 2 ..... 9
		step = init_step
		end = start + step * 10
		grids = np.arange(start, end, step)
		best_grid = self.tuner(terms, grids)
		#--------------------------------------
		# example : 5.5 6 6.5
		step = init_step / 2.
		start = best_grid[0] - step
		end = start + step * 2
		grids = np.arange(start, end, step)
		best_grid = self.tuner(terms, grids)
		#-----------------------------------------

	##############################################################################
	def tune_model_complexity(self,drop=''):
		'''
		This function can search for the suitable value for two parameters, in range [0-20]
		'''
		terms = ['max_depth', 'min_child_weight']
		#---------------------------------------------5
		pa1 = np.arange(1,10,2)
		pa2 = np.arange(1,10,2)
		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		bst = self.tuner(terms, grids)
		#-----------------------------------------------3
		pa1 = [ bst[0] - 1, bst[0], bst[0] + 1 ]
		pa2 = [ bst[1] - 1, bst[1], bst[1] + 1 ]
		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		bst = self.tuner(terms, grids)
		#--------------------------------------------------[5-7]
		if bst[0] < 10 and bst[1] < 10 :
			if bst[0] == pa1[0] : #5
				pa1 = np.arange(bst[0], bst[0]+1, 2)
			elif bst[0] == pa1[1] : #7
				pa1 = np.arange(bst[0]-.6, bst[0]+.8, .2)
			else : #5
				pa1 = np.arange(bst[0]-.8, bst[0]+.1, .2)

			if bst[1] == pa2[0] : #5
				pa2 = np.arange(bst[1], bst[1]+1, 2)
			elif bst[1] == pa2[1] : #7
				pa2 = np.arange(bst[1]-.6, bst[1]+.8, .2)
			else : #5
				pa2 = np.arange(bst[1]-.8, bst[1]+.1, .2)

			grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
			bst = self.tuner(terms, grids)
			#-------------------------------------------------3
			pa1 = [ bst[0]-.1, bst[0], bst[0]+.1 ]
			pa2 = [ bst[1]-.1, bst[1], bst[1]+.1 ]
			grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
			bst = self.tuner(terms, grids)
			#--------------------------------------------
			return bst
		#-------------------------------------------------------5/[5-7]
		# in case of either of them is 10.
		if bst[0] == 10 :
			pa1 = np.arange(11,20,2) # 5
		else :
			if bst[0] == pa1[0] : #5
				pa1 = np.arange(bst[0], bst[0]+1, 2)
			elif bst[0] == pa1[1] : #7
				pa1 = np.arange(bst[0]-.6, bst[0]+.8, .2)
			else : #5
				pa1 = np.arange(bst[0]-.8, bst[0]+.1, .2)

		if bst[1] == 10 : # 5
			pa2 = np.arange(11,20,2)
		else :
			if bst[1] == pa2[0] : #5
				pa2 = np.arange(bst[1], bst[1]+1, 2)
			elif bst[1] == pa2[1] : #7
				pa2 = np.arange(bst[1]-.6, bst[1]+.8, .2)
			else : #5
				pa2 = np.arange(bst[1]-.8, bst[1]+.1, .2)

		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		bst = self.tuner(terms, grids)
		#----------------------------------------------[5-7]/3
		if bst[0] >= 10 :
			if bst[0] == pa1[0] : #5
				pa1 = np.arange(bst[0], bst[0]+1, 2)
			elif bst[0] == pa1[1] : #7
				pa1 = np.arange(bst[0]-.6, bst[0]+.8, .2)
			else : #5
				pa1 = np.arange(bst[0]-.8, bst[0]+.1, .2)
		else :#3
			pa1 = [ bst[0]-.1, bst[0], bst[0]+.1 ]

		if bst[1] >= 10 :
			if bst[1] == pa2[0] : #5
				pa2 = np.arange(bst[1], bst[1]+1, 2)
			elif bst[1] == pa2[1] : #7
				pa2 = np.arange(bst[1]-.6, bst[1]+.8, .2)
			else : #5
				pa2 = np.arange(bst[1]-.8, bst[1]+.1, .2)
		else :#3
			pa2 = [ bst[1]-.1, bst[1], bst[1]+.1 ]

		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		bst = self.tuner(terms, grids)
		#---------------------------------------------------#3/1
		if bst[0] >= 10 : #3
			pa1 = [ bst[0]-.1, bst[0], bst[0]+.1 ]
		else :#1
			pa1 = [bst[0]]

		if bst[1] >= 10 :#3
			pa2 = [ bst[1]-.1, bst[1], bst[1]+.1 ]
		else :#1
			pa2 = [bst[1]]

		grids = [ (p1, p2) for p1 in pa1 for p2 in pa2 ]
		bst = self.tuner(terms, grids)
		#------------------------------------------------------
		return bst
	##############################################################
