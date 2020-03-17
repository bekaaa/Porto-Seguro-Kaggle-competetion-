import numpy as np
import pandas as pd
from xgb_tuner import xgb_tuner
import pickle
import log
params = {
    '''
    1 - Handling Imbalanced data.
	** max_delta_step and scale_pos_weight
	** scale_pos_weight is already assigned for each fold in xgb_tuner() with the ratio of
		negative examples to the positive examples.
	** max_delta_step not important to tune, but it should be a value from 0 to 10.
    '''
    'max_delta_step': 0,
    'scale_pos_weight': 1,  # calculated for each fold. #neg / #pos

    '''
    2 - Controling model complexity :
	I will tune them for a large scale of values.
    '''
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0,

    '''
    3 - control model's robusting to noise.
    '''
    'subsample': 1,
    'colsample_bytree': 1,

    '''Regularization'''
    'reg_alpha': 0,
    'reg_lambda': 1,

    'eta': 0.3,

    'objective': "binary:logistic",
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_seed': 0
}
seed = 0
log_index = 1
init_log(log_index)
log.msg('**********************')
log.msg('*** log file initialized ********')

with open('./data/dtrain.pkl', 'rb') as f:
    dtrain = pickle.load(f)
with open('./data/dvalid.pkl', 'rb') as f:
    dvalid = pickle.load(f)
log.msg('data has been loaded.')

rounds = 800
esrounds = 50 # early stop rounds.

tuner = xgb_tuner(dtrain, dvalid, params, logging=True, log_file_index=log_index,
                  rounds=rounds, esrounds=esrounds)
del dtrain, dvalid
log.msg('Tuning object has been loaded')



def init_log(index):
    log.LOG_PATH = './logs/'
    try:
        _ = log.close()
    except:
        pass
    log.init('tuning_params-' + str(index) + '.log')
def test_preds():
    with open('./data/dtest.pkl', 'rb') as f:
        dtest = pickle.load(f)
    return tuner.predict(dtest)


##############################################################
def tune_model_complexity():
    '''
    This function can search for the suitable int value for two parameters, in range [0-20]
    '''
    terms = ['max_depth', 'min_child_weight']
    all_results = []
    # ---------------------------------------------5
    pa1 = np.arange(1, 10, 2)
    pa2 = np.arange(1, 10, 2)
    grids = [(p1, p2) for p1 in pa1 for p2 in pa2]
    best_results, all_results = tuner(terms, grids, all_results)
    bst = best_results['grid']
    # -----------------------------------------------3
    pa1 = [bst[0] - 1, bst[0], bst[0] + 1]
    pa2 = [bst[1] - 1, bst[1], bst[1] + 1]
    grids = [(p1, p2) for p1 in pa1 for p2 in pa2]
    best_results, all_results = tuner(terms, grids, all_results)
    bst = best_results['grid']
    # --------------------------------------------------[Return]
    if bst[0] < 10 and bst[1] < 10:
        return terms, bst
    # -------------------------------------------------------5/[1]
    # in case of either of them is 10.
    if bst[0] == 10:
        pa1 = np.arange(11, 20, 2)  # 5
    else:
        pa1 = [bst[0]]

    if bst[1] == 10:  # 5
        pa2 = np.arange(11, 20, 2)
    else:
        pa2 = [bst[1]]

    grids = [(p1, p2) for p1 in pa1 for p2 in pa2]
    best_results, all_results = tuner(terms, grids, all_results)
    bst = best_results['grid']
    # ----------------------------------------------3/1
    if bst[0] >= 10:
        pa1 = [bst[0] - 1, bst[0], bst[0] + 1]
    else:  # 3
        pa1 = [bst[0]]

    if bst[1] >= 10:
        pa2 = [bst[1] - 1, bst[1], bst[1] + 1]
    else:  # 3
        pa2 = [bst[1]]

    grids = [(p1, p2) for p1 in pa1 for p2 in pa2]
    best_results, all_results = tuner(terms, grids, all_results)
    bst = best_results['grid']
    # ----------------------------------------------
    return terms, bst
##############################################################

terms, bst = tune_model_complexity()
for t, b in zip(terms, bst):
    print("*-* best %s = %g " % (t, b))
    log.msg("*-* best %s = %g " % (t, b))
    params[t] = b
print('\n')






