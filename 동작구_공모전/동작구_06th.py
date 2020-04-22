# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:33:32 2020

@author: ehdrb
"""
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import cross_val_score
from pyproj import Proj,transform
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

'''############데이터로드#################'''
sell_1 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Dongjak_xy_100000.csv')
sell_1.drop('Unnamed: 0',axis=1,inplace=True)
sell_2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Dongjak_xy_50000.csv')
sell_2.drop('Unnamed: 0',axis=1,inplace=True)

#accident
accident = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/accident.csv',encoding='cp949')
accident = accident[['사고유형구분','발생건수','위도','경도','UTM-K_경도변환','UTM-K_위도변환']] 
accident.rename(columns={'UTM-K_경도변환':'new_x','UTM-K_위도변환':'new_y'},inplace=True)
#child
child_place = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/child_place.csv')
#cw_point
crosswork = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/cw_point.csv',encoding='cp949')
crosswork.rename(columns={'X좌표':'new_x','Y좌표':'new_y'},inplace=True)
#old_place
old_place = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/old_place.csv')
#Bus
bus = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Bus.csv',encoding='cp949')
#Building
building = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Building.csv',encoding='cp949')
#bohang
bohang = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/bohang.csv',encoding='cp949')
#street_size
street_size = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/street_size.csv',encoding='cp949')
street_size=street_size.dropna()
street_size.isna().sum()

'''#############colum 제작##################'''
def make_col(sell,street_size=street_size):
    for col in street_size['도로규모'].unique():
        sell[col]=0
    return sell

'''###########유클리드 거리#################'''''
def count_facil(df_u,df_f,euclid,kind):
    name =[x for x in globals() if globals()[x] is df_u][0]
    
    for u_idx in df_u.index:
        num = 0
        
        if u_idx % 10000==0:
            print('현재 %s,%s의 %s 작업중입니다.'%(name,u_idx,kind))
            
        # 유클리드 거리 구하기
        distance = np.sqrt((df_u.loc[u_idx,'x'] - df_f['new_x'])**2 +
                           (df_u.loc[u_idx,'y'] - df_f['new_y'])**2)
        count= distance[distance<euclid]
            
            
        if kind =='accident' :
            for idx in count.index:
                num += len(count) * df_f.loc[idx,'발생건수']
        if kind =='building' :
            for idx in count.index:
                num += len(count) * df_f.loc[idx,'층수']
        if kind =='street_size':
            for idx in count.index:
                df_u.loc[u_idx,df_f.loc[idx,'도로규모']]+=1
                
        
        else:
            num = len(count)
            
        # count 값 삽입
        if kind !='street_size':
            df_u.loc[u_idx,kind]=num
        
    return df_u

'''###############유클리드 적용#####################''''
make_col(sell_1)
make_col(sell_2)

count_facil(sell_1,old_place,150,'old_place')
count_facil(sell_1,child_place,150,'child_place')
count_facil(sell_1,crosswork,150,'crosswork')
count_facil(sell_1,building,150,'building')
count_facil(sell_1,bus,150,'bus')
count_facil(sell_1,bohang,150,'bohang')
count_facil(sell_1,street_size,150,'street_size')
count_facil(sell_1,accident,150,'accident')
sell_1.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/sell_십만_150.csv')

sell_1 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Dongjak_xy_100000.csv')
sell_1.drop('Unnamed: 0',axis=1,inplace=True)
make_col(sell_1)

count_facil(sell_1,old_place,250,'old_place')
count_facil(sell_1,child_place,250,'child_place')
count_facil(sell_1,crosswork,250,'crosswork')
count_facil(sell_1,building,250,'building')
count_facil(sell_1,bus,250,'bus')
count_facil(sell_1,bohang,250,'bohang')
count_facil(sell_1,street_size,250,'street_size')
count_facil(sell_1,accident,250,'accident')
sell_1.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/sell_십만_250.csv')

#sell_2
count_facil(sell_2,old_place,150,'old_place')
count_facil(sell_2,child_place,150,'child_place')
count_facil(sell_2,crosswork,150,'crosswork')
count_facil(sell_2,building,150,'building')
count_facil(sell_2,bus,150,'bus')
count_facil(sell_2,bohang,150,'bohang')
count_facil(sell_2,street_size,150,'street_size')
count_facil(sell_2,accident,150,'accident')
sell_2.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/sell_오만_150.csv')

sell_2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Dongjak_xy_50000.csv')
sell_2.drop('Unnamed: 0',axis=1,inplace=True)
make_col(sell_2)

count_facil(sell_2,old_place,250,'old_place')
count_facil(sell_2,child_place,250,'child_place')
count_facil(sell_2,crosswork,250,'crosswork')
count_facil(sell_2,building,250,'building')
count_facil(sell_2,bus,250,'bus')
count_facil(sell_2,bohang,250,'bohang')
count_facil(sell_2,street_size,250,'street_size')
count_facil(sell_2,accident,250,'accident')
sell_2.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/sell_오만_250.csv')


'''############## Scailng, train,test split#################''''
robust_scaler = RobustScaler()
scaled_X=robust_scaler.fit_transform(sell[['old','child','crosswork']])
sell['old_scaled'] = scaled_X[:,0]
sell['child_scaled'] = scaled_X[:,1]
sell['crosswork_scaled'] = scaled_X[:,2]

#sell.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/sell_150.csv')


plt.hist(np.log1p(sell['accident']))
sell['accident_log']=np.log1p(sell['accident'])
'''##############모델링 ##########################''''
#1. OLS 다중회귀
model = smf.ols(formula = 'accident~child+old+crosswork',data=sell)
result = model.fit()
result.summary()

"""
Euclid =250
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               accident   R-squared:                       0.214
Model:                            OLS   Adj. R-squared:                  0.214
Method:                 Least Squares   F-statistic:                     4392.
Date:                Mon, 20 Apr 2020   Prob (F-statistic):               0.00
Time:                        23:13:26   Log-Likelihood:            -1.7223e+05
No. Observations:               48312   AIC:                         3.445e+05
Df Residuals:                   48308   BIC:                         3.445e+05
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      0.8148      0.067     12.231      0.000       0.684       0.945
child         -0.5894      0.016    -36.732      0.000      -0.621      -0.558
old            1.8607      0.063     29.407      0.000       1.737       1.985
crosswork      0.4746      0.004    105.616      0.000       0.466       0.483
==============================================================================
Omnibus:                    24507.382   Durbin-Watson:                   2.013
Prob(Omnibus):                  0.000   Jarque-Bera (JB):           171484.624
Skew:                           2.377   Prob(JB):                         0.00
Kurtosis:                      10.911   Cond. No.                         26.3
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""


