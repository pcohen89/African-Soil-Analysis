# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:25:30 2014

@author: p_cohen
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 03 09:56:37 2014

@author: p_cohen
"""
# import libraries that will be useful (superset of libraries needed for code)
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import  RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.ridge import RidgeCV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.linear_model import Ridge, RandomizedLasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import svm
import numpy as np
import time
import statsmodels as sm
from scipy.optimize import minimize
from __future__ import division
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.neighbors import KNeighborsRegressor


def replace_missings(data):
    # this replaces missings with medians
    # NOTE: mixed string num columns it does not do anything with
    for cols in data._get_numeric_data().columns:
        data[cols].fillna(value=data[cols].median(), inplace=True)

def build_data(data_pth, train_nm, cln_trn, test_nm, cln_test):
    # this function takes raw csvs and creates clean data frames,
    # this function does not create new variables
    rw_trn = pd.read_csv(data_pth+train_nm)
    rw_test = pd.read_csv(data_pth+test_nm)
    samps = (rw_trn, rw_test)
    for samp in samps:
        replace_missings(samp)         
    rw_trn.to_csv(data_pth + cln_trn)
    rw_test.to_csv(data_pth + cln_test)
 
def create_val_and_train(train, split_rate, seed):
    # this will create a test and train, BUT the randomization is at the 
    # LOCATION LEVEL, not the obs level
    ids = ['BSAN', 'BSAS', 'BSAV', 'CTI', 'ELEV', 'EVI', 'LSTD',
             'LSTN', 'REF1','REF2', 'REF3', 'REF7', 'RELI',
             'TMAP', 'TMFI']
    loc_dat = train[ids].drop_duplicates()
    np.random.seed(seed)
    rndm_vect = np.array(np.random.rand(len(loc_dat.ix[:,0]),1))
    loc_dat['rand_vals'] = (rndm_vect)
    train = pd.merge(train, loc_dat, on= ids)
    trn_trn = train[train['rand_vals'] > split_rate]
    trn_val = train[train['rand_vals'] <= split_rate]
    return trn_trn, trn_val
    
def lass_varselect(train, num_vars, target, alpha):   
    lass = RandomizedLasso(alpha=alpha, n_resampling=5)
    lass.fit(train[num_vars], train[target])
    return lass.get_support()

def write_preds(models, name, train, val, test, outcome):
    train[name] = 0
    val[name] = 0
    test[name] = 0
    for mod in models:
        train[name] += train[mod]/len(models)
        val[name] += val[mod]/len(models)
        test[name] += test[mod]/len(models) 
        MSE = ((val[mod] - val[outcome]) ** 2).mean()
        RMSE = np.sqrt(MSE)
        print mod + " has RMSE of " + str(object=RMSE)
        
def pred_sand(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    sand_lassoed_vars = lass_varselect(train, all_vars, 'Sand', .00000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1200)
    univ_selector.fit(train[all_vars], train['Sand'])
    pvals = univ_selector.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if sand_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if sand_lassoed_vars[x]:
            lass_only.append(all_vars[x]) 

    # randomforest
    forst = RandomForestRegressor(n_estimators=100)
    forst.fit(train.ix[:, chosen], train['Sand'])
    for dset in data:
        dset['sand_for_prds'] = forst.predict(dset.ix[:, chosen])
        
    # SVM
    svr = svm.SVR(C=14, epsilon=.43, kernel='linear')
    svr.fit(train.ix[:, lass_only], train['Sand'])
    for dset in data:
        dset['sand_svr_prds'] = svr.predict(dset.ix[:, lass_only])
        
    # lasso
    lass = Lasso(alpha=.0000001, positive=True)
    lass.fit(train[all_vars], train['Sand'])
    for dset in data:
        dset['sand_las_prds'] = lass.predict(dset[all_vars])

    # ridge
    sand_ridge = RidgeCV(np.array([.7]), normalize=True)
    sand_ridge.fit(train[all_vars], train['Sand'])
    for dset in data:
        dset['sand_rdg_prds'] = sand_ridge.predict(dset[all_vars])
    # combination
    models= ['sand_las_prds', 'sand_rdg_prds', 
             'sand_for_prds', 'sand_for_prds', 'sand_svr_prds'] 
    #print train.ix[0:20, models]
    name = 'sand_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'Sand')
    
def pred_SOC(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    SOC_lassoed_vars = lass_varselect(train, all_vars, 'SOC', .000000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1200)
    univ_selector.fit(train[all_vars], train['SOC'])
    pvals = univ_selector.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if SOC_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if SOC_lassoed_vars[x]:
            lass_only.append(all_vars[x])   
    #randomforest
    neigh = RandomForestRegressor(n_estimators=100)
    neigh.fit(train.ix[:, chosen], train['SOC'])
    for dset in data:
        dset['SOC_for_prds'] = neigh.predict(dset.ix[:, chosen])   
    # lasso
    lass = Lasso(alpha=.00000025, positive=True)
    lass.fit(train[all_vars], train['SOC'])
    for dset in data:
        dset['SOC_las_prds'] = lass.predict(dset[all_vars])
    # ridge
    SOC_ridge = RidgeCV(np.array([.5]), normalize=True)
    SOC_ridge.fit(train[all_vars], train['SOC'])
    for dset in data:
        dset['SOC_rdg_prds'] = SOC_ridge.predict(dset[all_vars])
    # combination
    models= ['SOC_las_prds', 'SOC_rdg_prds', 
              'SOC_for_prds', 'SOC_for_prds' ] 
    name = 'SOC_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'SOC')
    
def pred_pH(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    pH_lassoed_vars = lass_varselect(train, all_vars, 'pH', .00000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1200)
    univ_selector.fit(train[all_vars], train['pH'])
    pvals = univ_selector.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if pH_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if pH_lassoed_vars[x]:
            lass_only.append(all_vars[x])
    # nearest randomforest
    neigh = RandomForestRegressor(n_estimators=100)
    neigh.fit(train.ix[:, chosen], train['pH'])
    for dset in data:
        dset['pH_for_prds'] = neigh.predict(dset.ix[:, chosen])  
    # lasso
    lass = Lasso(alpha=.000000275, positive=True)
    lass.fit(train[all_vars], train['pH'])
    for dset in data:
        dset['pH_las_prds'] = lass.predict(dset[all_vars])
    # ridge
    pH_ridge = RidgeCV(np.array([.6]), normalize=True)
    pH_ridge.fit(train[all_vars], train['pH'])
    for dset in data:
        dset['pH_rdg_prds'] = pH_ridge.predict(dset[all_vars])
    # combination
    models= [ 'pH_rdg_prds', 'pH_las_prds', 
              'pH_for_prds', 'pH_for_prds' ] 
    name = 'pH_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'pH')
    
def pred_P(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    P_lassoed_vars = lass_varselect(train, all_vars, 'P', .00000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1600)
    univ_selector.fit(train[all_vars], train['P'])
    pvals = univ_selector.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if P_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if P_lassoed_vars[x]:
            lass_only.append(all_vars[x])
    chosen.append('sand_prds' + str(object=loop))
    chosen.append('pH_prds' + str(object=loop))
    chosen.append('SOC_prds' + str(object=loop))
    chosen.append('Ca_prds' + str(object=loop))
    # SVM
    svr = svm.SVR(C=10000, epsilon=.1)
    svr.fit(train.ix[:, all_vars], train['P'])
    for dset in data:
        dset['P_svr_prds'] = svr.predict(dset.ix[:, all_vars])
  
    gbr = GradientBoostingRegressor(n_estimators = 60,
        learning_rate = 0.1, max_depth =5, random_state = 42, 
        verbose = 0, min_samples_leaf=4)
    gbr.fit(train.ix[:, chosen], train['P'])
    for dset in data:
        dset['P_gbr_prds'] = gbr.predict(dset.ix[:,chosen])
    # ridge
    P_ridge = RidgeCV(np.array([.55]), normalize=True)
    P_ridge.fit(train[all_vars], train['P'])
    for dset in data:
        dset['P_rdg_prds'] = P_ridge.predict(dset[all_vars])
    # combination
    models= [ 'P_rdg_prds', 
              'P_svr_prds', 'P_gbr_prds'] #, 'P_las_prds' , 'P_gbr_prds'
    name = 'P_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'P')
    
def pred_Ca(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    Ca_lassoed_vars = lass_varselect(train, all_vars, 'Ca', .0000000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1400)
    univ_selector.fit(train[all_vars], train['Ca'])
    pvals = univ_selector.get_support()
    chosen =  []
    for x in range(1, len(all_vars)):
        if Ca_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if Ca_lassoed_vars[x]:
            lass_only.append(all_vars[x])
    # nearest randomforest
    forst = RandomForestRegressor(n_estimators=120)
    forst.fit(train.ix[:, chosen], train['Ca'])
    #print forst.feature_importances_
    for dset in data:
        dset['Ca_for_prds'] = forst.predict(dset.ix[:, chosen])
        
    # lasso
    lass = Lasso(alpha=.0000001, positive=True)
    lass.fit(train[all_vars], train['Ca'])
    for dset in data:
        dset['Ca_las_prds'] = lass.predict(dset[all_vars])
    # ridge
    Ca_ridge = RidgeCV(np.array([.5]), normalize=True)
    Ca_ridge.fit(train[all_vars], train['Ca'])
    for dset in data:
        dset['Ca_rdg_prds'] = Ca_ridge.predict(dset[all_vars])
    # combination
    models= ['Ca_las_prds', 'Ca_rdg_prds', 
             'Ca_for_prds', 'Ca_for_prds',  ] 
    name = 'Ca_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'Ca')
       
def run_models(data_pth, cln_trn, cln_test, targets, subm_path, subm_nm, seed,
               outcome_frame, loops):
    trn = pd.read_csv(data_pth+cln_trn)
    test = pd.read_csv(data_pth+cln_test)
    trn['soil_type'] = 0
    trn['soil_type'][trn['Depth'] == 'Topsoil'] = 1
    test['soil_type'] = 0
    test['soil_type'][test['Depth'] == 'Topsoil'] = 1
    all_vars = test._get_numeric_data().columns
    for x in range(3, len(all_vars)-6):
        name = "diff_" + str(object=all_vars[x])
        trn[name] = trn.ix[:, x] - trn.ix[:, x+1]
        test[name] = test.ix[:, x] - test.ix[:, x+1]
    all_vars = test._get_numeric_data().columns     
    preds = [ 'Ca_prds', 'P_prds',  'pH_prds', 'SOC_prds', 'sand_prds' ] 
    actuals = ['Ca', 'P', 'pH', 'SOC', 'Sand']
    for num in range(0,len(preds)):
        test[preds[num]] = 0
    for x in range(0, loops):
        t0= time.time() 
        print "Loop: " + str(object=x)
        trn_trn, trn_val = create_val_and_train(trn, .001, x)
        trn_trn['P'][trn_trn['P']>3] = 3 + (trn_trn['P'] - 3) * .3
        trn_trn['P'][trn_trn['P']<-3] = -3 - (trn_trn['P'] + 3) * .3
       
        ## predicting Sand ##
        pred_sand(trn_trn, trn_val, test, all_vars, x)       
        ## predicting SOC ##
        pred_SOC(trn_trn, trn_val, test, all_vars, x)        
        ## predicting pH ##
        pred_pH(trn_trn, trn_val, test, all_vars, x)       
        ## predicting Ca ##
        pred_Ca(trn_trn, trn_val, test, all_vars, x)        
        ## predicting P ##
        pred_P(trn_trn, trn_val, test, all_vars, x) 
        ### Evaluate models ####               
        for num in range(0,len(preds)):
            name = preds[num] + str(object=x)
            MSE = ((trn_val[name] - trn_val[actuals[num]]) ** 2).mean()
            RMSE = np.sqrt(MSE)
            print name + " has RMSE of " + str(object=RMSE)
            outcome_frame.ix[x, actuals[num]] = RMSE
            test[preds[num]] += test[name]/(loops)
        title = "It took {time} minutes to run loop"
        print title.format(time=(time.time()-t0)/60)
    return test
                      
################## Importing data ###################

kaggle_pth = "S:\General\Training\Ongoing Professional Development\Kaggle/"
data_pth = "Predicting Soil Properties\Data\Raw/"
sum_pth = "Predicting Soil Properties\Data\Summary stats/"
subm_path = "Predicting Soil Properties\Data\Submissions/"
targets = ['Ca', 'P', 'pH', 'SOC', 'Sand']
predictions = [ 'Ca_prds', 'P_prds',  'pH_prds', 'SOC_prds', 'sand_prds' ] 
full_path = kaggle_pth + data_pth

iterations = 1
zer = np.zeros(iterations)
out_dict = {'Sand' : zer, 'SOC' : zer, 'pH' : zer, 'P' : zer, 'Ca': zer}
outcomes = pd.DataFrame(out_dict)
test = run_models(full_path, "cln_trn.csv", "cln_test.csv", targets, 
                  subm_path, "first submission", 46, outcomes, iterations) 
outcomes.mean()
outcomes.mean().mean()

submission = pd.DataFrame(test['PIDN'])
for vars in range(0, len(targets)):
    submission[targets[vars]] = test[predictions[vars]]
sub_nm = 'abhi pure replication.csv'  
submission.to_csv(kaggle_pth + subm_path + sub_nm, index=False)
