#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#uncomment if needed:
#pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
import math
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import time


# In[3]:


def imputation_time(dataframe, feature_matrix, n_est, dataset_name):
    """Measure and plot the imputation time for both tested estimators.
    
    Keyword arguments:
    dataframe -- The dataframe containing missing values
    feature_matrix -- The dataframe storing the results of 
    the descriptive analysis
    n_est -- Number of trees in the Random Forest
    dataset_name -- Name of dataframe
    
    Output:
    heatmap of imputation time for both estimators.
        
    """
    timing_RF=[]
    timing_BR=[]
    
    m_list=[5, 10, 15, 20]
    mis_fraction_list=[0.05, 0.1, 0.2, 0.35, 0.5]
    
    m_count=[]
    mis_count=[]
    
    for m in m_list:
        for i in mis_fraction_list:
            rnd_numbers_column, rnd_numbers_row=generate_rnd(dataframe, i)
            data_missing, error_i, feature_matrix=create_missing(
                dataframe, rnd_numbers_row, rnd_numbers_column, feature_matrix, dataset_name)
            
            t1 = time.perf_counter()
            
            impute_RandomForest(dataframe, data_missing, rnd_numbers_row, rnd_numbers_column, error_i, m,  n_est)
            t2 = time.perf_counter()
            
            impute_BayesRegression(dataframe, data_missing, rnd_numbers_row, rnd_numbers_column, error_i, m)
            t3 = time.perf_counter()
            
            timing_RF.append(t2-t1)
            timing_BR.append(t3-t2)
            
            m_count.append(m)
            mis_count.append(i)
            
    time_matrix=pd.DataFrame()
    time_matrix['m']=m_count
    time_matrix['mis_fraction']=mis_count
    time_matrix['time elapsed RF']=timing_RF
    time_matrix['time elapsed BR']=timing_BR
    time_matrix_pivot_RF = time_matrix.pivot('m', 'mis_fraction', 'time elapsed RF')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4.5))
    sns.heatmap(time_matrix_pivot_RF, vmin=0, cmap='rainbow', ax=ax1, cbar_kws={
        "shrink": .75, 'label': 'time elapsed in seconds'})
            
    time_matrix_pivot_BR = time_matrix.pivot('m', 'mis_fraction', 'time elapsed BR')
            
    sns.heatmap(time_matrix_pivot_BR, vmin=0, cmap='rainbow', ax=ax2,cbar_kws={
        "shrink": .75, 'label': 'time elapsed in seconds'}) 
            
    ax1.set_title('Random Forest', weight='bold').set_fontsize('18')
    ax2.set_title('Bayesian Ridge', weight='bold').set_fontsize('18')
    ax1.axis('equal')
    ax2.axis('equal')
    
def feature_weights(feature_matrix):
    """Obtain and plot relative influence of each feature on imputation performance.
    
    Keyword arguments:
    feature_matrix -- The dataframe storing the results of 
    the descriptive analysis  
    
    Output:
    barplot displaying size of each feature's regression coefficient.
    
    """
    plt.figure(figsize=(8,5))
    
    #Get weight of dataset features in classification of estimators
    feature_matrix_lr=feature_matrix.copy(deep=True)
    feature_matrix_lr['Best estimator'].replace('No difference', np.nan, inplace=True)
    feature_matrix_lr.dropna(inplace=True)
    feature_matrix_lr.reset_index(drop=True, inplace=True)
    X=feature_matrix_lr.iloc[:,2:16]
    y=feature_matrix_lr.iloc[:,16]
    y.replace('Random Forest',1,  inplace=True)
    y.replace('Bayesian Regression',0, inplace=True)
    scaler=StandardScaler()
    clf_BR = BayesianRidge(normalize=False).fit(scaler.fit_transform(X),y)

    pred_coef=clf_BR.coef_
    pred_coef=pd.DataFrame(pred_coef)
    pred_coef.columns=['Regression Coefficient']
    pred_coef.index=feature_matrix.columns[2:16]
    model_indicator=[]
    index_new=[]
    for coef in pred_coef['Regression Coefficient'].values:
        if coef < 0:
            model_indicator.append('Bayesian Ridge')
        else:
            model_indicator.append('Random Forest')
       
    pred_coef['Association with model performance']=model_indicator
    pred_coef=pred_coef.sort_values(by='Regression Coefficient', axis=0, ascending=False) 
    g=sns.barplot(x=pred_coef.index, y='Regression Coefficient', data=pred_coef, dodge=False,palette=['red', 'blue'], 
                  hue='Association with model performance', ci=None)
    plt.setp(g.get_xticklabels(), rotation=90)
    g

def performance_overall(result_matrix):
    """Compare and plot performance of estimators for all 
    variations of meta-features and datasets
    
    Keyword arguments:
    result_matrix -- The dataframe storing the results of 
    the comparative imputation analysis 
    
    Output:
    Lineplot comparing imputation performance of both estimators 
    over all datasets including a barplot displaying the relative
    differences in imputation performance [unit: NRMSE].
        
    """
    plt.figure(figsize=(10,6))
    
    xticks=[]
    for x, pval in enumerate(result_matrix['p (Mann-Whitney-U)'].values):
        if pval < 0.01:
            xticks.append(str(x+1)+'**')
        elif pval < 0.05:
            xticks.append(str(x+1)+'*')
        else:
            xticks.append(str(x+1))
        
    a=sns.lineplot(x=np.arange(0,len(result_matrix['Dataset']),1), y=result_matrix['NRMSE RF'].astype(np.float), 
                   data=result_matrix, color='red')
    sns.lineplot(x=np.arange(0,len(result_matrix['Dataset']),1), y=result_matrix['NRMSE BR'].astype(np.float), 
                 data=result_matrix, color='blue')
    ax2 = plt.twinx()
    b=sns.barplot(x=xticks, y=result_matrix['NRMSE RF'].astype(np.float)-result_matrix['NRMSE BR'].astype(np.float), 
                  data=result_matrix, alpha=.3 , color='firebrick', ax=ax2)

    a.text(7.2, 0.115, 'Bayesian Ridge', horizontalalignment='left', verticalalignment='center', color='darkblue')
    a.text(7.2, 0.109, 'Random Forest', horizontalalignment='left', verticalalignment='center', color='red')

    rect = patches.Rectangle((6.75,0.105),5.5,0.015,linewidth=1,edgecolor='black',facecolor='white')
    a.add_patch(rect)

    b.axhline(0, color='black', linewidth=0.5)

    a.set_xlabel('Datasets')
    a.set_ylabel('NRMSE')
    b.set_ylabel('Δ NRMSE (RF vs. BR)')

    plt.margins(0.025)
    
def performance_per_pct_missing(result_matrix_joined):
    """#Compare performance of estimators for each Dataset over all pct_missings
    
    Keyword arguments:
    result_matrix_joined -- combined dataset of analyses for all variations of 
    missing data fractions
    
    Output:
    Separate lineplots for each dataset comparing imputation performance
    of both estimators over severall missing value percentages.
    
    """
    rows=math.ceil(len(set(result_matrix_joined['Dataset']))/3)
    fig, axes = plt.subplots(rows,3 , figsize=(18,rows*4))

    dataset_RF=[]
    dataset_BR=[]
    num_df=len(set(result_matrix_joined['Dataset']))
    max_df=result_matrix_joined.shape[0]//num_df
    for d in range(num_df):
        for e in range(max_df):
            dataset_RF.append(result_matrix_joined['NRMSE RF'][d+(num_df*e)])
            dataset_BR.append(result_matrix_joined['NRMSE BR'][d+(num_df*e)])

    pct_missing=[2,5,10,20, 35, 50]
    j=0
    k=0
    m=np.arange(max_df,result_matrix_joined.shape[0]+1,max_df)
    for i, n in enumerate(m):
        o=n-max_df
        sns.lineplot(x=pct_missing, y=dataset_RF[o:n], data=result_matrix_joined, color='red',ax=axes[j][k]
                    ).set_title(result_matrix_joined['Dataset'][i], weight='bold').set_fontsize('18')
        sns.lineplot(x=pct_missing, y=dataset_BR[o:n], data=result_matrix_joined, ax=axes[j][k]
                    ).set(xlabel='pct_missing', ylabel='NRMSE')

        max_val=max(max(zip(dataset_BR[o:n], dataset_RF[o:n])))
        axes[j][k].text(2, max_val-(max_val/20), 'Bayesian Ridge', horizontalalignment='left', 
                        verticalalignment='center', color='darkblue')
        axes[j][k].text(2, max_val-(max_val/8), 'Random Forest', horizontalalignment='left', 
                        verticalalignment='center', color='red')
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        axes[j][k].set_xlabel('Missing values [%]')
        k=k+1
        if k/3%1==0 and i!=0:
            j=j+1
            k=0


# In[ ]:




