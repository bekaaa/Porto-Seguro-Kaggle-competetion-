import numpy as np
import pandas as pd
from xgb_tuner import xgb_tuner
import pickle
from logger import logger
params = {
    'max_delta_step': 0,
    'scale_pos_weight': 1,  # calculated for each fold. #neg / #pos
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 1,
    'colsample_bytree': 1,
    'reg_alpha': 0,
    'reg_lambda': 1,
    'eta': 0.3,
    'objective': "binary:logistic",
    'eval_metric': 'auc',
    'n_jobs': -1,
    'random_seed': 0
}
seed = 0
# initializing logger object
log_index = 1
log = logger('tuning_params-' + str(log_index) + '.log','./logs')
log.add('**********************')
log.add('*** log file initialized ********')

# loading data
with open('./data/dtrain.pkl', 'rb') as f:
    dtrain = pickle.load(f)
with open('./data/dvalid.pkl', 'rb') as f:
    dvalid = pickle.load(f)
log.add('data has been loaded.')

rounds = 800
esrounds = 50 # early stop rounds.
# Tuner object
tuner = xgb_tuner(dtrain, dvalid, params,
                  rounds=rounds, esrounds=esrounds,nfolds=3, log=log)
del dtrain, dvalid
log.add('Tuning object has been loaded')


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
    log.add("*-* best %s = %g " % (t, b))
    params[t] = b
print('\n')






