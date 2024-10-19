#what things does this code do?
#1 split train, validation, test set based on the whole vector data matrix.
#2 use five-time validation and optuna to find the optimal parameters
#3 get the AUROC and AUPRC during the five-time validation with the optimal parameters, to see the model performace
#4 train the final model using the optimal params based on training and validation set, ann use test set to see model performance


import polars as pl
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve,auc


def get_train_validation_test_matrix(data_tuple,res=False,smote=False,use_test=False): #return(train_fx,train_fy,val_fx,val_fy)或者(tv_fx,tv_fy,test_fx,test_fy) 
    if use_test == False:
        train_x, train_y, val_x, test_y = data_tuple
    elif use_test == True:
        tv_x, tv_y, test_x, test_y = data_tuple

    if use_test == False:
        if res == True: #if resample the compounds
            sampling_strategy="not majority"
            ros=RandomOverSampler(sampling_strategy=sampling_strategy,random_state=2)
            train_x_res,train_y_res = ros.fit_resample(pd.DataFrame(train_x),train_y)

            #Obtain the fingerprints corresponding to the list of compounds
            ltrain=sum(train_x_res.values.tolist(),[]) #Merge nested lists into a single list
            df_set_train=pl.DataFrame()
            for smi in ltrain:
                df_smi=df.filter(pl.col('stan_smiles')==smi)
                df_set_train=pl.concat([df_set_train,df_smi],how='vertical')
        elif res == False:
            ltrain0=list(train_x)
            df_set_train=df.filter(pl.col('stan_smiles').is_in(ltrain0))

        if smote == True:
            smote = SMOTE(random_state=3)
            train_fx=df_set_train[:,4:]
            train_fy=df_set_train[:,3]
            train_fx, train_fy = smote.fit_resample(np.array(train_fx), train_fy)
        elif smote == False:
            train_fx=df_set_train[:,4:]
            train_fy=df_set_train[:,3]

        df_set_val=df.filter(pl.col('stan_smiles').is_in(list(val_x)))
        val_fx=df_set_val[:,4:]
        val_fy=df_set_val[:,3]
        # print(train_fy.value_counts(),val_fy.value_counts())
        return(train_fx,train_fy,val_fx,val_fy)
    elif use_test == True: #After obtaining the optimal parameters, the whole of the training and validation sets are trained, and use test set to see the model performance
        if res == True:
            sampling_strategy="not majority"
            ros=RandomOverSampler(sampling_strategy=sampling_strategy,random_state=2)
            tv_x_res,tv_y_res = ros.fit_resample(pd.DataFrame(tv_x),tv_y) 

            ltv=sum(tv_x_res.values.tolist(),[])
            df_set_tv=pl.DataFrame()
            for smi in ltv:
                df_smi=df.filter(pl.col('stan_smiles')==smi)
                df_set_tv=pl.concat([df_set_tv,df_smi],how='vertical')
        elif res == False:
            ltv0=list(tv_x)
            df_set_tv=df.filter(pl.col('stan_smiles').is_in(ltv0))

        if smote == True:
            smote = SMOTE(random_state=3)
            tv_fx=df_set_tv[:,4:]
            tv_fy=df_set_tv[:,3]
            tv_fx, tv_fy = smote.fit_resample(np.array(tv_fx), tv_fy)
        elif smote == False:
            tv_fx=df_set_tv[:,4:]
            tv_fy=df_set_tv[:,3]

        df_set_test=df.filter(pl.col('stan_smiles').is_in(list(test_x)))
        test_fx=df_set_test[:,4:]
        test_fy=df_set_test[:,3]
        return(tv_fx,tv_fy,test_fx,test_fy)

def xgb_model(train_x,train_y,test_x,test_y,params,model_path):
    model=XGBClassifier(booster='gbtree',tree_method='gpu_hist',objective='binary:logistic',**params)
    model.fit(train_x, train_y)
    model.save_model(model_path)
    ypred = model.predict_proba(test_x)[:,1]
    return ypred

def get_values(test_y,ypred,threshold=0.5): #print test results
    precision, recall, thresholds = precision_recall_curve(test_y, ypred)
    print('AUPRC: %.4F' %auc(recall,precision))
    print ('AUROC: %.4f' % metrics.roc_auc_score(test_y,ypred))

def objective(trial):
    param = {
        'booster':'gbtree',
        'tree_method':'gpu_hist', 
        "objective": "binary:logistic",
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 256),
        'gamma': trial.suggest_float('gamma', 1e-7, 10.0,log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0,log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0,log=True),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
    }
    fiveval_scores=np.zeros(5)

    for idx,(train_fx,train_fy,test_fx,test_fy) in enumerate(fiveval):
        model = XGBClassifier(**param)
        model.fit(train_fx, train_fy)
        y_pred = model.predict_proba(test_fx)[:,1]
        fiveval_scores[idx] = roc_auc_score(test_fy,y_pred)

    return np.mean(fiveval_scores) #AUROC


assaylist=['0_Aromatase_stansmi_activity_spectrumid_vector.csv','1_AhR_stansmi_activity_spectrumid_vector.csv',
           '2_AR_stansmi_activity_spectrumid_vector.csv','3_ER_stansmi_activity_spectrumid_vector.csv',
           '4_GR_stansmi_activity_spectrumid_vector.csv','5_TSHR_stansmi_activity_spectrumid_vector.csv',
           '6_TR_stansmi_activity_spectrumid_vector']

for idx0,assay in enumerate(assaylist):
    print('assay = ', assay)
    df=pl.read_csv(os.path.join(r"7assays_spectrumid_vectors",assay))  #path of assays' data

    #=============
    df1=df[:,2:4].unique(keep='first',maintain_order=True) #Take all non-repeating rows to get the compound-activity list，maintain_order=True is to remain their original order.
    X=df1[:,0]; Y=df1[:,1]

    if assay=='3_ER_stansmi_activity_spectrumid_vector.csv' or assay=='4_GR_stansmi_activity_spectrumid_vector.csv':
        a=5
    else:
        a=7

    #The list of compounds are first divided 3:1:1, using random, stratified sampling
    tv_x, test_x, tv_y, test_y = train_test_split(X,Y,test_size=0.2,random_state=a,stratify=Y,shuffle=True) #The random state that divides the test set is fixed for a specific assay
                                                                                                            #random_state=5 for ER GR, and random_state=5 for others
    #split validating datasets using StratifiedShuffleSplit
    #=============
    fiveval=[]
    sss=StratifiedShuffleSplit(n_splits=5,train_size=0.75,test_size=0.25,random_state=2)
    for train_idx,val_idx in sss.split(tv_x,tv_y):
        train_x, train_y = tv_x[train_idx], tv_y[train_idx]
        val_x, val_y = tv_x[val_idx], tv_y[val_idx]
        data_tuple=(train_x,train_y,val_x,val_y)
        fiveval.append(get_train_validation_test_matrix(data_tuple,res=True,smote=True,use_test=False))
    fiveval

    #to find the optimal params
    #=============
    import optuna
    study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler())
    n_trials=50 
    study.optimize(objective, n_trials=n_trials)
    print('Number of finished trials:', len(study.trials))
    print("------------------------------------------------")
    print('Best trial:', study.best_trial.params)
    print("------------------------------------------------")

    param=study.best_params
    # print(params)

    #to get five AUROC and AUPRC values when training with optimal params
    #use the optimal params to retrain the training set, and evaluate on validation set (five times)
    #=============
    auroc_fiveval_scores=np.zeros(5)
    auprc_fiveval_scores=np.zeros(5)

    for idx,(train_fx,train_fy,val_fx,val_fy) in enumerate(fiveval):
        model = XGBClassifier(booster='gbtree',tree_method='gpu_hist',objective='binary:logistic',**param)
        model.fit(train_fx, train_fy)
        y_pred = model.predict_proba(val_fx)[:,1]
        auroc_fiveval_scores[idx] = roc_auc_score(val_fy,y_pred)
        precision, recall, thresholds = precision_recall_curve(val_fy, y_pred)
        auprc_fiveval_scores[idx] =auc(recall,precision)
    
    print('AUROC')
    print(auroc_fiveval_scores)
    print('auroc_five_time_validation_mean',np.mean(auroc_fiveval_scores))

    print('AUPRC')
    print(auprc_fiveval_scores)
    print('auprc_five_time_validation_mean',np.mean(auprc_fiveval_scores))

    #use training and validation sets to train the optimal model and use the test set to see performance 
    #=============
    tv_fx,tv_fy,test_fx,test_fy=get_train_validation_test_matrix((tv_x,tv_y,test_x,test_y),res=True,smote=True,use_test=True)
    model_path=os.path.join(r"models for 7 assays",assay.split('.csv')[0]+'.model') #the path to save the final optimal model
    ypred=xgb_model(tv_fx,tv_fy,test_fx,test_fy,param,model_path) 
    get_values(test_fy,ypred)
    print('\n=======================\n')
