#!/usr/bin/env python
# coding: utf-8

# In[32]:


#uncomment if needed:
#!conda install -c conda-forge colored
#pip install --upgrade scikit-learn

import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randint
import math
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
import colored
from colored import stylize


# In[34]:


def create_matrices():
    feature_matrix_dict={'Dataset':'','pct_missing':'','n':'', 'Attributes':'','ratio_attributes to instances':'','pct_metric':'', 'pct_binary':'', 'pct_categorical':'','Mean of numeric values':'','Std of numeric values':'','Avg distinct values of categorical variables':'','pct_outliers':'','pct_normally distributed attributes':'', 'AVG Skewness':'','AVG Kurtosis':'','pct_highly correlated variables (>.4)':'', 'Best estimator': ''}
    feature_matrix=pd.DataFrame(feature_matrix_dict, index=[0]) 
    result_matrix_dict={'Dataset':'', 'NRMSE RF':'', 'NRMSE BR':'', 'Between-variance (z-score) RF':'', 'Between-variance (z-score) BR':'', 'p (Mann-Whitney-U)':''}
    result_matrix=pd.DataFrame(result_matrix_dict, index=[0])
    return feature_matrix, result_matrix

def update_analysis_matrix(dataset_name, analysis_matrix, feature_name, value):
    if analysis_matrix['Dataset'][0]=='':
        analysis_matrix['Dataset'][0]=dataset_name        
    if dataset_name in analysis_matrix['Dataset'].values:
        if feature_name=='pct_missing' and analysis_matrix['pct_missing'].iloc[-1]!=value:
            if analysis_matrix['pct_missing'].iloc[-1]=='':
                analysis_matrix.iloc[np.where(analysis_matrix['Dataset'] == dataset_name)[0][-1],
                                     np.where(analysis_matrix.columns == feature_name)[0][-1]]=value
            else:
                blank_row=pd.DataFrame(np.zeros([1,analysis_matrix.shape[1]]), columns=analysis_matrix.columns)
                blank_row.iloc[0,0]=dataset_name
                analysis_matrix = pd.concat([analysis_matrix, blank_row], ignore_index=True)
                analysis_matrix.iloc[np.where(analysis_matrix['Dataset'] == dataset_name)[0][-1],
                                     np.where(analysis_matrix.columns == 'pct_missing')[0][-1]]=value
        else:
            analysis_matrix.iloc[np.where(analysis_matrix['Dataset'] == dataset_name)[0][-1],
                                 np.where(analysis_matrix.columns == feature_name)[0][-1]]=value
    else:
        blank_row=pd.DataFrame(np.zeros([1,analysis_matrix.shape[1]]), columns=analysis_matrix.columns)
        blank_row.iloc[0,0]=dataset_name
        analysis_matrix = pd.concat([analysis_matrix, blank_row], ignore_index=True)
        analysis_matrix.iloc[np.where(analysis_matrix['Dataset'] == dataset_name)[0][-1],
                             np.where(analysis_matrix.columns == feature_name)[0][-1]]=value
    return analysis_matrix

def get_means_mean_std(dataframe, dataset_name, feature_matrix):
    dtype_df=pd.DataFrame(dataframe.dtypes)
    mean_temp=0
    std_temp=0
    c=0
    for col, dtype in zip(dtype_df.index, dtype_df.values):
        if dtype[0]=='float64':
            mean_temp=mean_temp+np.mean(dataframe[col].values)
            std_temp=std_temp+np.std(dataframe[col].values)
            c=c+1
        else:
            pass
    try:
        mean_of_means=float("{0:.2f}".format(mean_temp/c))
    except:
        mean_of_means=0
    try:
        mean_of_std=float("{0:.2f}".format(std_temp/c))
    except:
        mean_of_std=0
    update_analysis_matrix(dataset_name, feature_matrix, 'Mean of numeric values', mean_of_means)
    update_analysis_matrix(dataset_name, feature_matrix, 'Std of numeric values', mean_of_std)
    
def get_outliers(dataframe, dataset_name, feature_matrix):
    outlier_score=0
    for col in dataframe.columns:
        quant1, quant3 = np.percentile(dataframe[col], [25 ,75])
        iqr = quant3 - quant1
        #Multiply the IQR with 1.5 as in boxplots to get the upper and lower limit
        #Any value outside of these limits is an outlier
        upper_limit=quant3+iqr*1.5
        lower_limit=quant1-iqr*1.5
        for val in dataframe[col]:
            if val < lower_limit or val > upper_limit:
                outlier_score=outlier_score+1
            else:
                pass
    update_analysis_matrix(dataset_name, feature_matrix, 'pct_outliers', float("{0:.2f}".format(
        outlier_score/(dataframe.shape[0]*dataframe.shape[1])*100)))
    
def get_datatype(dataset_name, dataframe, feature_matrix):
    numeric_count=0
    binary_count=0
    categorical_count=0
    distinct_cat=[]
    num_check=0
    info=pd.DataFrame(dataframe.dtypes.value_counts())
    for i in range(len(info)):
        if (info.index[i]=='float64' or info.index[i]=='int64') and num_check==0:
            num_check=1
            for col in dataframe.columns:
                if len(dataframe[col].value_counts())<3:
                    binary_count=binary_count+1
                else:
                    num_marker=0
                    for value in dataframe[col].values:
                        if value-int(value)==0:
                            pass
                        else:
                            num_marker=num_marker+1
                    if num_marker!=0:
                        numeric_count=numeric_count+1
                    elif len(set(dataframe[col]))>15:
                        numeric_count=numeric_count+1
                    else:
                        categorical_count=categorical_count+1
                        distinct_cat.append(len(set(dataframe[col])))
        elif info.index[i]=='object':
            for t, val in enumerate(dataframe.dtypes.values):
                if val =='object':
                    distinct_cat.append(len(set(basic_data_df.iloc[:,t])))                    
            categorical_count=categorical_count+info.values[i][0]
        elif num_check!=0:
            pass
        else:
            print(dataset_name,'Contains unknown datatype',info.index[i],'. Please check.')
    update_analysis_matrix(dataset_name, feature_matrix, 'pct_metric', 
                          float("{0:.2f}".format((numeric_count/dataframe.shape[1])*100)))
    update_analysis_matrix(dataset_name, feature_matrix, 'pct_binary', 
                          float("{0:.2f}".format((binary_count/dataframe.shape[1])*100)))
    update_analysis_matrix(dataset_name, feature_matrix, 'pct_categorical', 
                          float("{0:.2f}".format((categorical_count/dataframe.shape[1])*100))) 
    if len(distinct_cat)!=0:
        update_analysis_matrix(dataset_name, feature_matrix, 
                              'Avg distinct values of categorical variables',
                              np.sum(distinct_cat)//len(distinct_cat))
    else:
        update_analysis_matrix(dataset_name, feature_matrix, 
                              'Avg distinct values of categorical variables', 0)
    get_means_mean_std(dataframe, dataset_name, feature_matrix)
    get_outliers(dataframe, dataset_name, feature_matrix)
    
def get_high_cor(dataframe, dataset_name, feature_matrix):
    corr_mat=dataframe.corr()
    col_count=0
    high_cor_count=0
    for col in corr_mat.columns:
        for i, val in enumerate(corr_mat[col]):
            #do not count correlations (r=1) with self
            if i==col_count:
                pass
            else:
                if val >= 0.4:
                    high_cor_count=high_cor_count+1
        col_count=col_count+1
    pct_high_cor=float("{0:.2f}".format(((high_cor_count/2)/
                                         ((pow(dataframe.shape[1],2)-dataframe.shape[1])/2))*100))
    update_analysis_matrix(dataset_name, feature_matrix, 
                          'pct_highly correlated variables (>.4)', pct_high_cor)
    
def get_skewness(dataframe, dataset_name, feature_matrix):
    skew=pd.DataFrame(dataframe.skew())
    skew.columns=['Skewness']
    color_indicator_skew=[]
    skew_sum=0
    for skewness in skew.values:
        if skewness <-1 or skewness >1:
            color_indicator_skew.append('highly skewed')
        elif skewness >=-1 and skewness < -0.5 or skewness >0.5 and skewness <=1:
            color_indicator_skew.append ('moderately skewed')
        else:
            color_indicator_skew.append ('approximately symmetric')
        skew_sum=skew_sum+math.sqrt(pow(skewness,2))
    skew_avg=float("{0:.4f}".format(skew_sum/skew.shape[0]))
    update_analysis_matrix(dataset_name, feature_matrix, 'AVG Skewness', skew_avg)
    plt.figure(figsize=(8,5))
    s=sns.barplot(x=skew.index, y='Skewness', data=skew, hue=color_indicator_skew, palette=['g', 'r', 'b'])
    s.set_xticklabels(s.get_xticklabels(), rotation=90)
    s
    
def get_kurtosis(dataframe, dataset_name, feature_matrix):
    kurtosis=pd.DataFrame(dataframe.kurtosis())
    kurtosis.columns=['Kurtosis']
    color_indicator_kurt=[]
    kurt_sum=0
    for kurt in kurtosis.values:
        if kurt <-1 or kurt >1:
            color_indicator_kurt.append('too peaked or flat')
        elif kurt >=-1 and kurt < -0.5 or kurt >0.5 and kurt <=1:
            color_indicator_kurt.append ('moderately peaked or flat')
        else:
            color_indicator_kurt.append ('approximately normal-shaped')
        kurt_sum=kurt_sum+math.sqrt(pow(kurt,2))
    kurt_avg=float("{0:.4f}".format(kurt_sum/kurtosis.shape[0]))
    update_analysis_matrix(dataset_name, feature_matrix, 'AVG Kurtosis', kurt_avg)
    plt.figure(figsize=(8,5))
    k=sns.barplot(x=kurtosis.index, y='Kurtosis', data=kurtosis, hue=color_indicator_kurt, palette=['r', 'g', 'b'])
    k.set_xticklabels(k.get_xticklabels(), rotation=90)
    k
        
def normal_test(dataset_name, dataframe, feature_matrix, alpha=5e-3):
    not_normal_list=[]
    normal_list=[]
    for i in range(dataframe.shape[1]):
        p=stats.normaltest(dataframe[dataframe.columns[i]]).pvalue
        if p<alpha:
            not_normal_list.append(dataframe.columns[i])
        else:
            normal_list.append(dataframe.columns[i])
    update_analysis_matrix(dataset_name, feature_matrix, 'pct_normally distributed attributes', 
                          float("{0:.2f}".format(len(normal_list)/
                                                 (len(normal_list)+len(not_normal_list))*100)))
    update_analysis_matrix(dataset_name, feature_matrix, 'Attributes', dataframe.shape[1])
    update_analysis_matrix(dataset_name, feature_matrix, 'ratio_attributes to instances', 
                          float("{0:.4f}".format(dataframe.shape[1]/dataframe.shape[0])))
    update_analysis_matrix(dataset_name, feature_matrix, 'n', dataframe.shape[0])
    get_high_cor(dataframe, dataset_name, feature_matrix)
    print('Columns',not_normal_list,'are',stylize('not', colored.attr("bold")),'normally distributed!\n')
    print('Columns',normal_list,'are normally distributed!')
    plot_normaldist(dataframe)
        
def plot_normaldist(dataframe):
    rows=math.ceil(dataframe.shape[1]/4)
    fig, axes = plt.subplots(rows, 4, figsize=(18, rows*4))
    j=0
    k=0
    for i in range(1, dataframe.shape[1]+1):
        if rows <2:
            ax=axes[k]
        else:
            ax=axes[j][k]
        sns.distplot(dataframe[dataframe.columns[i-1]], hist=False, ax=ax).set_title(dataframe.columns[i-1], weight='bold').set_fontsize('12')
        ax.set_xlabel('')
        k=k+1
        if i/4%1==0:
            j=j+1
            k=0
            
def generate_rnd(dataframe, mis_fraction=0.02):
    seed(9)    
    rnd_numbers_column = randint(0, dataframe.shape[1], 
                                 round(dataframe.shape[0]*dataframe.shape[1]*mis_fraction))
    rnd_numbers_row=randint(0,dataframe.shape[0], 
                            round(dataframe.shape[0]*dataframe.shape[1]*mis_fraction))
    return rnd_numbers_column, rnd_numbers_row

def create_missing(dataframe, rnd_numbers_row, rnd_numbers_column, feature_matrix, dataset_name):
    df_missing=dataframe.copy()
    error_count=0
    error_i=[]
    for i, row in enumerate(rnd_numbers_row):
        if pd.isnull(df_missing.iloc[row, rnd_numbers_column[i]]):
            error_count=error_count+1
            error_i.append(i)
        else:
            df_missing.iloc[row, rnd_numbers_column[i]]=np.nan
    n_missing=len(rnd_numbers_row)-error_count
    percentage_missing = float("{0:.2f}".format((n_missing/(dataframe.shape[0]*dataframe.shape[1]))*100))
    feature_matrix=update_analysis_matrix(dataset_name, feature_matrix, 'pct_missing', percentage_missing)
    return df_missing, error_i, feature_matrix

def impute_RandomForest(dataframe, df_missing, rnd_numbers_row, rnd_numbers_column, error_i, m,  n_est):
    imputed_value_temp_RF=pd.DataFrame()
    imputed_value_list_RF=[]
    for i in range(m):
        imp_RF = IterativeImputer(tol=0.01, max_iter=10, random_state=i, 
                                  sample_posterior=False, 
                                  estimator=RandomForestRegressor(bootstrap=True, 
                                                                  max_samples=0.4, n_jobs=-1, 
                                                                  min_impurity_decrease=1e-4, 
                                                                  n_estimators=n_est))
        df_imputed=pd.DataFrame(imp_RF.fit_transform(df_missing))
        for k, row in enumerate(rnd_numbers_row):
            if k in error_i:
                pass
            else:
                imputed_value_list_RF.append(df_imputed.iloc[row, rnd_numbers_column[k]])
        imputed_value_temp_RF[i]=imputed_value_list_RF
        imputed_value_list_RF=[]
    df_imputed.columns=[df_missing.columns.tolist()]
    return df_imputed, imputed_value_temp_RF

def impute_BayesRegression(dataframe, df_missing, rnd_numbers_row, rnd_numbers_column, error_i, m):
    imputed_value_temp=pd.DataFrame()
    imputed_value_list=[]
    for i in range(m):
        imp_BR = IterativeImputer(tol=0.01, max_iter=10, sample_posterior=True, 
                                  estimator=BayesianRidge(normalize=True, alpha_1=0, lambda_1=0.005))
        df_imputed=pd.DataFrame(imp_BR.fit_transform(df_missing))
        for k, row in enumerate(rnd_numbers_row):
            if k in error_i:
                pass
            else:
                imputed_value_list.append(df_imputed.iloc[row, rnd_numbers_column[k]])            
        imputed_value_temp[i]=imputed_value_list
        imputed_value_list=[] 
    df_imputed.columns=[df_missing.columns.tolist()]
    return df_imputed, imputed_value_temp

def compare_imp(dataframe, imputed_value_temp, rnd_numbers_row, rnd_numbers_column, error_i, estimator):
    compare_imp_dict={'True_value':'', 'Imputed_value':''}
    true_value_list=[]
    imputed_value_list=imputed_value_temp.mean(axis=1).tolist()
    imputed_df_final=dataframe.copy(deep=True)
    j=0
    for i, row in enumerate(rnd_numbers_row):
        if i in error_i:
                pass
        else:
            true_value_list.append(dataframe.iloc[row, rnd_numbers_column[i]])
            imputed_df_final.iloc[row, rnd_numbers_column[i]]=imputed_value_list[j]
            j=j+1
    compare_imp_dict['True_value']=true_value_list
    compare_imp_dict['Imputed_value']=imputed_value_list
    compare_imp_df=pd.DataFrame(compare_imp_dict)
    print(stylize("Imputed values - {}: \n".format(estimator), colored.attr("bold")))
    print(compare_imp_df.head(),'\n\n')
    return compare_imp_df, imputed_df_final

def scatterplot_imp(compare_RF, compare_Bayes):
    imp_sorted_RF=compare_RF.sort_values(by='True_value')
    imp_sorted_Bayes=compare_Bayes.sort_values(by='True_value')
    
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    sns.lineplot(np.arange(max(imp_sorted_RF['True_value'])+1),
                 np.arange(max(imp_sorted_RF['True_value'])+1), color='coral', lw=3,ax=axes[0])
    sns.lineplot(np.arange(max(imp_sorted_RF['True_value'])+1),
                 np.arange(max(imp_sorted_RF['True_value'])+1), color='coral', lw=3,ax=axes[1])
    sns.scatterplot(imp_sorted_RF['True_value'], imp_sorted_RF['Imputed_value'], 
                    ax=axes[0], s=200, alpha=0.5).set_title('Random Forest')
    sns.scatterplot(imp_sorted_Bayes['True_value'], imp_sorted_Bayes['Imputed_value'], 
                    ax=axes[1], color='c', s=200, alpha=0.5).set_title('Bayesian Regression')

def distplot_imp(compare_RF, compare_Bayes):
    fig, axes = plt.subplots(1, 2, figsize=(13,6))
    imp_sorted_RF=compare_RF.sort_values(by='True_value')
    imp_sorted_Bayes=compare_Bayes.sort_values(by='True_value')
    
    ax=sns.distplot(imp_sorted_RF['True_value'], hist=False, color='r', 
                    label='True value', ax=axes[0])
    ax1=sns.distplot(imp_sorted_RF['Imputed_value'], hist=False, color='b', 
                     ax=ax, label='Imputed value').set_title('Random Forest')
    
    ax2=sns.distplot(imp_sorted_Bayes['True_value'], hist=False, color='r', 
                     label='True value', ax=axes[1])
    ax3=sns.distplot(imp_sorted_Bayes['Imputed_value'], hist=False, color='b', 
                     ax=ax2, label='Imputed value').set_title('Bayesian Regression')

def calc_nrmse(dataframe, imputed_df_final, estimator):
    nrmse_list=[]
    for i in range(imputed_df_final.shape[1]):
        rmse_temp=math.sqrt(np.sum((pow(imputed_df_final.iloc[:, i]-dataframe.iloc[:,i],2)/dataframe.shape[0])))
        try:
            nrmse_temp=rmse_temp/(max(dataframe.iloc[:, i])-min(dataframe.iloc[:, i]))
        except:
            nrmse_temp=rmse_temp/max(dataframe.iloc[:, i])
        nrmse_list.append(nrmse_temp)
    nrmse=np.sum(nrmse_list)/len(nrmse_list)
    print(stylize('NRMSE', colored.attr("bold")),'for {}:'.format(estimator), nrmse)
    return nrmse, nrmse_list

def add_final_scores(dataset_name, nrmse_BR, nrmse_RF, m_whitney, feature_matrix, result_matrix):
    #threshold for significance alpha=10% as two-sided test is used
    if m_whitney>-0.1 and m_whitney<0.1:
        if nrmse_BR>nrmse_RF:
            update_analysis_matrix(dataset_name, feature_matrix, 'Best estimator', 'Random Forest')
        else:
            update_analysis_matrix(dataset_name, feature_matrix, 'Best estimator', 'Bayesian Regression')
    else:
        update_analysis_matrix(dataset_name, feature_matrix, 'Best estimator', 'No difference')
    update_analysis_matrix(dataset_name, result_matrix, 'NRMSE RF', nrmse_RF)
    update_analysis_matrix(dataset_name, result_matrix, 'NRMSE BR', nrmse_BR)
    update_analysis_matrix(dataset_name, result_matrix, 'p (Mann-Whitney-U)', m_whitney)
    
def calc_between_var(dataset_name, matrix_imputed, estimator, result_matrix):
    z_score_list=[]
    for i in range(len(matrix_imputed)):
        z_scores=0
        row_mean=np.mean((matrix_imputed.iloc[i,:]))
        std=np.std(matrix_imputed.iloc[i,:])
        if std!=0:
            for j in range(len(matrix_imputed.iloc[i,:])):
                z_scores=z_scores+abs((matrix_imputed.iloc[i,j]-row_mean)/std)
        else:
            z_scores=0
        z_score_list.append(z_scores/len(matrix_imputed.iloc[i,:]))
    z_score_final=np.mean(z_score_list)
    result_matrix=update_analysis_matrix(dataset_name, result_matrix, 
                                        'Between-variance (z-score) {}'.format(estimator),  z_score_final)
    return result_matrix 


# In[ ]:




