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
    
def general_data_summary(samp, num_smp, samp_nm):
    print "Obs in " + samp_nm + ": " + str(object=samp.iloc[:, 0].count())
    print "Cols in " + samp_nm + ": " + str(object=samp.iloc[0, :].count())
    num_cols = num_smp.iloc[0, :].count()
    print "Num cols in " + samp_nm + ": " + str(object=num_cols)
    
def browse_data(data_pth, cln_trn, cln_test, target, output_path):
    trn = pd.read_csv(data_pth+cln_trn)
    test = pd.read_csv(data_pth+cln_test)
    num_trn = trn._get_numeric_data()
    num_tst = test._get_numeric_data()
    ##### general summary stats of data set ####
    general_data_summary(trn, num_trn, "train")
    general_data_summary(test, num_tst, "test")
    #### summary stats of variables ####
    # means
    trn_out = pd.DataFrame(num_trn.mean().reset_index())
    tst_out = pd.DataFrame(num_tst.mean().reset_index())
    trn_out.columns = ['id', 'mean']
    tst_out.columns = ['id', 'mean']
    # stds
    trn_std_data = pd.DataFrame(num_trn.std().reset_index())
    tst_std_data = pd.DataFrame(num_tst.std().reset_index())
    trn_std_data.columns = ['id', 'std']
    tst_std_data.columns = ['id', 'std']
    # correlation w/ outcome (train only)
    trn_std_data['targetcor'] = 0
    trn_std_data['abs_targetcor'] = 0
    for var in range(1, len(num_tst.columns)): # use tst to exclude targets
        correl = num_trn[[trn_std_data['id'][var], target]].corr().iloc[0,1]
        trn_std_data.ix[var, 'targetcor'] = correl
        trn_std_data.ix[var, 'abs_targetcor'] = abs(correl)
    # merging
    trn_out = pd.merge(trn_out, trn_std_data, on= 'id')
    tst_out = pd.merge(tst_out, tst_std_data, on= 'id')
    # exporting
    trn_out.to_csv(kaggle_pth + output_path + "training summary.csv")
    tst_out.to_csv(kaggle_pth + output_path + "test summary.csv")
 
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
   
    # nearest nieghbors
    #neigh = KNeighborsRegressor(n_neighbors=2)
    #neigh.fit(train.ix[:, chosen], train['Sand'])
    #for dset in data:
      #  dset['sand_ngh_prds'] = neigh.predict(dset.ix[:, chosen])
        
    # SVM
    #svr = svm.SVR()
    #svr.fit(train.ix[:, lass_only], train['Sand'])
    #for dset in data:
        #dset['sand_svr_prds'] = svr.predict(dset.ix[:, lass_only])
    # randomforest
    forst = RandomForestRegressor(n_estimators=200)
    forst.fit(train.ix[:, chosen], train['Sand'])
    for dset in data:
        dset['sand_for_prds'] = forst.predict(dset.ix[:, chosen])
        
    # SVM
    svr = svm.SVR(C=23000)
    svr.fit(train.ix[:, all_vars], train['Sand'])
    for dset in data:
        dset['sand_svr_prds'] = svr.predict(dset.ix[:, all_vars])
        
    # lasso
    #lass = Lasso(alpha=.0000001, positive=True)
    #lass.fit(train[all_vars], train['Sand'])
    #for dset in data:
    #    dset['sand_las_prds'] = lass.predict(dset[all_vars])

    # ridge
    sand_ridge = RidgeCV(np.array([1.135]), normalize=True)
    sand_ridge.fit(train[all_vars], train['Sand'])
    for dset in data:
        dset['sand_rdg_prds'] = sand_ridge.predict(dset[all_vars])
    # combination
    models= [ 'sand_rdg_prds', 'sand_svr_prds',
             'sand_for_prds',  'sand_svr_prds'] 
    #print train.ix[0:20, models]
    name = 'sand_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'Sand')
    
def pred_SOC(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    SOC_lassoed_vars = lass_varselect(train, all_vars, 'SOC', .000000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 4500)
    univ_selector.fit(train[all_vars], train['SOC'])
    univ_selector2 = SelectKBest(score_func = f_regression, k = 200)
    univ_selector2.fit(train[all_vars], train['SOC'])
    pvals = univ_selector.get_support()
    pvals2 = univ_selector2.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if SOC_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    chosen2 =  []
    for x in range(0, len(all_vars)):
        if SOC_lassoed_vars[x] | pvals2[x]:
            chosen2.append(all_vars[x])
    lass_only =  []
    for x in range(0, len(all_vars)):
        if SOC_lassoed_vars[x]:
            lass_only.append(all_vars[x])    
    #randomforest
    forst = RandomForestRegressor(n_estimators=120)
    forst.fit(train.ix[:, chosen], train['SOC'])
    for dset in data:
        dset['SOC_for_prds'] = forst.predict(dset.ix[:, chosen])
    gbr = GradientBoostingRegressor(n_estimators = 900,
            learning_rate = .0785, max_depth =1, random_state = 42, 
            verbose = 0, min_samples_leaf=4, subsample = .4)
    gbr.fit(train[chosen2], train['SOC'])
    for dset in data:
        dset['SOC_gbr_prds'] = gbr.predict(dset.ix[:, chosen2])    
    # lasso
    #lass = Lasso(alpha=.00000025, positive=True)
    #lass.fit(train[all_vars], train['SOC'])
    #for dset in data:
    #    dset['SOC_las_prds'] = lass.predict(dset[all_vars])

    # ridge
    SOC_ridge = RidgeCV(np.array([.315]), normalize=True)
    SOC_ridge.fit(train[all_vars], train['SOC'])
    for dset in data:
        dset['SOC_rdg_prds'] = SOC_ridge.predict(dset[all_vars])
    # SVR
    svr = svm.SVR(C=9000, epsilon=.1)
    svr.fit(train.ix[:, chosen], train['SOC'])
    for dset in data:
        dset['SOC_svr_prds'] = svr.predict(dset.ix[:, chosen])
    # combination
    models= ['SOC_rdg_prds', 'SOC_svr_prds',
              'SOC_gbr_prds', 'SOC_for_prds',  'SOC_svr_prds' ]
    name = 'SOC_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'SOC')
    
def pred_pH(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    pH_lassoed_vars = lass_varselect(train, all_vars, 'pH', .00000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 1200) # intentionally unchanged
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
    # nearest nieghbors
    #neigh = KNeighborsRegressor(n_neighbors=4)
    #neigh.fit(train.ix[:, chosen], train['pH'])
    #for dset in data:
        #dset['pH_ngh_prds'] = neigh.predict(dset.ix[:, chosen])    
    # nearest randomforest
    forst = RandomForestRegressor(n_estimators=200)
    forst.fit(train.ix[:, chosen], train['pH'])
    for dset in data:
        dset['pH_for_prds'] = forst.predict(dset.ix[:, chosen])  
    # lasso
    #lass = Lasso(alpha=.000000275, positive=True)
    #lass.fit(train[all_vars], train['pH'])
    #for dset in data:
    #    dset['pH_las_prds'] = lass.predict(dset[all_vars])
    # ridge
    pH_ridge = RidgeCV(np.array([.6]), normalize=True)
    pH_ridge.fit(train[all_vars], train['pH'])
    for dset in data:
        dset['pH_rdg_prds'] = pH_ridge.predict(dset[all_vars])
    svr = svm.SVR(C=11000, epsilon=.1)
    svr.fit(train.ix[:, all_vars], train['pH'])
    for dset in data:
        dset['pH_svr_prds'] = svr.predict(dset.ix[:, all_vars])
    # combination
    models= [ 'pH_rdg_prds', 'pH_svr_prds', 'pH_svr_prds',
             'pH_for_prds' ] 
    name = 'pH_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'pH')
    
def pred_P(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    P_lassoed_vars = lass_varselect(train, all_vars, 'P', .0000000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 6450)
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
    # SVM
    svr = svm.SVR(C=15000, epsilon=.1)
    svr.fit(train.ix[:, chosen], train['P'])
    for dset in data:
        dset['P_svr_prds'] = svr.predict(dset.ix[:, chosen])
    # Gradient boosting regression
    gbr = GradientBoostingRegressor(n_estimators = 60,
        learning_rate = 0.1, max_depth =5, random_state = 42, 
        verbose = 0, min_samples_leaf=4)
    gbr.fit(train.ix[:, all_vars], train['P'])
    for dset in data:
        dset['P_gbr_prds'] = gbr.predict(dset.ix[:, all_vars])
    # ridge
    P_ridge = RidgeCV(np.array([3.95]), normalize=True)
    P_ridge.fit(train[all_vars], train['P'])
    for dset in data:
        dset['P_rdg_prds'] = P_ridge.predict(dset[all_vars])
    # combination
    models= [ 'P_rdg_prds', 
              'P_svr_prds', 'P_gbr_prds'] 
    name = 'P_prds' + str(object=loop)
    write_preds(models, name, train, val, test, 'P')
    
def pred_Ca(train, val, test, all_vars, loop):
    data = (val, test, train)
    # variable selection
    Ca_lassoed_vars = lass_varselect(train, all_vars, 'Ca', .0000000001)
    univ_selector = SelectKBest(score_func = f_regression, k = 5000)
    univ_selector.fit(train[all_vars], train['Ca'])
    univ_selector2 = SelectKBest(score_func = f_regression, k = 200)
    univ_selector2.fit(train[all_vars], train['Ca'])
    pvals = univ_selector.get_support()
    pvals2 = univ_selector2.get_support()
    chosen =  []
    for x in range(0, len(all_vars)):
        if Ca_lassoed_vars[x] | pvals[x]:
            chosen.append(all_vars[x])
    chosen2 =  []
    for x in range(0, len(all_vars)):
        if Ca_lassoed_vars[x] | pvals2[x]:
            chosen2.append(all_vars[x])
    gbr = GradientBoostingRegressor(n_estimators = 1000,
        learning_rate = .1695, max_depth =1, random_state = 42, 
        verbose = 0, min_samples_leaf=4)
    gbr.fit(train[chosen2], train['Ca'])
    for dset in data:
       dset['Ca_gbr_prds'] = gbr.predict(dset.ix[:, chosen2])
    # nearest randomforest
    forst = RandomForestRegressor(n_estimators=120)
    forst.fit(train.ix[:, chosen], train['Ca'])
    for dset in data:
        dset['Ca_for_prds'] = forst.predict(dset.ix[:, chosen])
        
    # ridge
    Ca_ridge = RidgeCV(np.array([4.925]), normalize=True)
    Ca_ridge.fit(train[all_vars], train['Ca'])
    for dset in data:
        dset['Ca_rdg_prds'] = Ca_ridge.predict(dset[all_vars])
    # SVR model
    svr = svm.SVR(C=9500)
    svr.fit(train.ix[:, chosen], train['Ca'])
    for dset in data:
        dset['Ca_svr_prds'] = svr.predict(dset.ix[:, chosen])

    # combination
    models= [ 'Ca_rdg_prds', 'Ca_gbr_prds',  
              'Ca_for_prds', 'Ca_svr_prds', 'Ca_svr_prds' ]   
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
    for x in range(3, len(all_vars)-1):
       name = "diff_" + str(object=all_vars[x])
       trn[name] = trn.ix[:, x] - trn.ix[:, x+1]
       test[name] = test.ix[:, x] - test.ix[:, x+1]
    all_vars = test._get_numeric_data().columns
    preds =  [  'SOC_prds',  ] 
    actuals =  [ 'SOC', ]
    for num in range(0,len(preds)):
        test[preds[num]] = 0
    
    for x in range(0, loops):
        t0= time.time() 
        print "Loop: " + str(object=x)
        trn_trn, trn_val = create_val_and_train(trn, .2, x)
        trn_trn['P'][trn_trn['P']>3] = 3 + (trn_trn['P'] - 3) * .3
        trn_trn['P'][trn_trn['P']<-3] = -3 - (trn_trn['P'] + 3) * .3       
        ## predicting Sand ##
        #pred_sand(trn_trn, trn_val, test, all_vars, x)       
        ## predicting SOC ##
        pred_SOC(trn_trn, trn_val, test, all_vars, x)        
        ## predicting pH ##
        #pred_pH(trn_trn, trn_val, test, all_vars, x)        
        ## predicting Ca ##
        #pred_Ca(trn_trn, trn_val, test, all_vars, x)        
        ## predicting P ##
        #pred_P(trn_trn, trn_val, test, all_vars, x)
        
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
# ['Ca', 'P', 'pH', 'SOC', 'Sand']
#[ 'Ca_prds', 'P_prds',  'pH_prds', 'SOC_prds', 'sand_prds' ]
predictions = [ 'Ca_prds', 'P_prds',  'pH_prds', 'SOC_prds', 'sand_prds' ] 
full_path = kaggle_pth + data_pth

build_data(full_path, "training.csv", "cln_trn.csv",
           "sorted_test.csv", "cln_test.csv" )
#browse_data(full_path, "cln_trn.csv", "cln_test.csv", "Sand", sum_pth)

iterations = 15
zer = np.zeros(iterations)
out_dict = {'Sand' : zer, 'SOC' : zer, 'pH' : zer, 'P' : zer, 'Ca': zer}
outcomes = pd.DataFrame(out_dict)
test = run_models(full_path, "training.csv", "sorted_test.csv", targets, 
                  subm_path, "first submission", 46, outcomes, iterations) 
outcomes.mean()
outcomes.mean().mean()

submission = pd.DataFrame(test['PIDN'])
for vars in range(0, len(targets)):
    submission[targets[vars]] = test[predictions[vars]]
sub_nm = 'from old starting point adding GBRs selectively.csv'  
submission.to_csv(kaggle_pth + subm_path + sub_nm, index=False)

###### Play code ##################################
bh = pd.read_csv(full_path+"training.csv")
ch = pd.read_csv(full_path+"sorted_test.csv")
bh['soil_type'] = 0
bh['soil_type'][bh['Depth'] == 'Topsoil'] = 1
ch['soil_type'] = 0
ch['soil_type'][ch['Depth'] == 'Topsoil'] = 1
all_vars = ch._get_numeric_data().columns 
for x in range(3, len(all_vars)-1):
    name = "diff_" + str(object=all_vars[x])
    bh[name] = bh.ix[:, x] - bh.ix[:, x+1]
    ch[name] = ch.ix[:, x] - ch.ix[:, x+1]
all_vars = ch._get_numeric_data().columns  

def train_SOC( init_guess):
    RMSE = 0
    loop_start = 1
    loop_end = 15
    print "Running loop with C = " + str(object=init_guess)
    for loop in range(loop_start, loop_end):
        t0= time.time()
        bh_trn, bh_val = create_val_and_train(bh, .2, loop)
        bh_trn['P'][ bh_trn['P']>3] = 3 + ( bh_trn['P'] - 3) * .3
        bh_trn['P'][ bh_trn['P']<-3] = -3 - ( bh_trn['P'] + 3) * .3
        all_vars = ch._get_numeric_data().columns
        data = [bh_trn, bh_val]
        #print init_guess
        # variable selection
        SOC_lassoed_vars = lass_varselect(bh_trn, all_vars, 'SOC', .000000001)
        univ_selector = SelectKBest(score_func = f_regression, k = 200)
        univ_selector.fit(bh_trn[all_vars], bh_trn['SOC'])
        pvals = univ_selector.get_support()
        chosen =  []
        for x in range(0, len(all_vars)):
            if SOC_lassoed_vars[x] | pvals[x]:     
                chosen.append(all_vars[x])
        lass_only =  []
        for x in range(0, len(all_vars)):
            if SOC_lassoed_vars[x]:
                lass_only.append(all_vars[x])
        print "Lass only: " + str(object=len(lass_only))       
        #forst = RandomForestRegressor(n_estimators=400, max_depth = 3)
        #forst.fit(bh_trn[chosen], bh_trn['SOC'])
        #for dset in data:
        #    dset['SOC_for_prds'] = forst.predict(dset.ix[:, chosen])
        #SOC_ridge = RidgeCV(np.array([.375]), normalize=True)
        #SOC_ridge.fit(bh_trn[all_vars], bh_trn['SOC'])
        #for dset in data:
        #    dset['SOC_rdg_prds'] = SOC_ridge.predict(dset[all_vars])
        #svr = svm.SVR(C=15000, epsilon = init_guess)
        #svr.fit(bh_trn[chosen], bh_trn['SOC'])
        #for dset in data:
        #   dset['SOC_svr_prds'] = svr.predict(dset.ix[:, chosen])
        gbr = GradientBoostingRegressor(n_estimators = 900,
            learning_rate = init_guess, max_depth =1, random_state = 42, 
            verbose = 0, min_samples_leaf=4, subsample = .7)
        gbr.fit(bh_trn[chosen], bh_trn['SOC'])
        for dset in data:
           dset['SOC_gbr_prds'] = gbr.predict(dset.ix[:, chosen])
        
          # combination
        
        title = "It took {time} minutes to run loop " + str(object=loop)
        print title.format(time=(time.time()-t0)/60)
        loop_RMSE = np.sqrt(((bh_val['SOC_gbr_prds'] - bh_val['SOC']) ** 2).mean())
        print "RMSE of loop is " + str(object=loop_RMSE)
        RMSE += loop_RMSE/(loop_end-loop_start)
    print RMSE
    return RMSE
        
init_guess = np.array([.09]) 
optimizer = minimize(train_SOC, init_guess,
                     method='nelder-mead' , options= {'xtol':.1, 'disp':True})    

for x in range(1, 2):
    y = .070 + x*.003
    train_SOC()
    
train_SOC(.08)