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
    'objective': 'reg:linear', # reg:logistic, binary:logistic, multi:softmax
    'eval_metric' : 'error', # error for binary class., merror for multiclass classification.
    'seed' : 0
}