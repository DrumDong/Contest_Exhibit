# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:58:53 2020

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
proj_WGS84 = Proj(init='epsg:4326') # epsg:4326
proj_UTMK = Proj(init='epsg:5186')
def transform_w84_to_utmk(df):
    return pd.Series(transform(proj_WGS84,proj_UTMK,df['위치정보(X)'],df['위치정보(Y)']),index=['위치정보(X)','위치정보(Y)'])


---데이터 로드----
old=pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/old_place.csv')
old.drop('Unnamed: 0',axis=1,inplace=True)
old.dropna(axis=0,inplace=True)
old=old.reset_index()
old.drop('index',axis=1,inplace=True)

child1 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/동작구_어린이집_1(UTM_K).csv',encoding='cp949')
child2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/동작구_유치원_1.csv(UTM_K).csv',encoding='cp949')
child1=child1[['소재지도로명주소','lat','lon','UTM-K_경도변환','UTM-K_위도변환']]
child2=child2[['주소','lat','lon','UTM-K_경도변환','UTM-K_위도변환']]
child1.columns = ['주소','위도','경도','new_x','new_y']
child2.columns = ['주소','위도','경도','new_x','new_y']
child=pd.concat([child1,child2])
child=child.reset_index()
child.drop('index',axis=1,inplace=True)
child.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/child.csv',encoding='cp949')

crosswork = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/cw_point.csv',encoding='cp949')
crosswork.rename(columns={'X좌표':'new_x','Y좌표':'new_y'},inplace=True)

accident = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/accident.csv',encoding='cp949')
accident = accident[['사고유형구분','발생건수','위도','경도','UTM-K_경도변환','UTM-K_위도변환']] 
accident.rename(columns={'UTM-K_경도변환':'new_x','UTM-K_위도변환':'new_y'},inplace=True)

---------예행연습----------
# 좌: 37.486606, 126.904874(경도) #191586.93794685882, 543021.8150669667
# 우: 37.483377, 126.981928(경도) #198401.62118190178, 542659.3419727073
# 위: 37.513634(위도), 126.942822 #194944.92916181506, 546018.8584748491
# 아래: 37.475563(위도), 126.971006 #197435.35736556537, 541792.3337019808

#임의 유동인구 csv 제작
updown=[random.randint(194944,197435) for r in range(10000)]
side=[random.randint(542659,543021) for r in range(10000)]
Udong = pd.DataFrame({'X':updown,'Y':side})

# 노인데이터, 어린이데이터를 Udong인구에 결합.
Udong.describe() # 50으로 기준 잡아볼까?

#-----------count 함수화----------------
def count_facil(df_u,df_f,euclid,kind):
    for u_idx in df_u.index:
        print(u_idx)
        num=0
        
        #시설 개수 count
        
        for f_idx in df_f.index:
            
            
            # 유클리드 거리 구하기
            cha=np.sqrt((df_u.loc[u_idx,'X']- df_f.loc[f_idx,'new_x'])**2+
                        (df_u.loc[u_idx,'Y']- df_f.loc[f_idx,'new_y'])**2)
            if kind =='accident':
                if cha < euclid:
                    num+=1*df_f.loc[f_idx,'발생건수']
            else:
                if cha < euclid:
                    num+=1
                
        # count 값 삽입
        Udong.loc[u_idx,kind]=num
        
    return Udong

count_facil(Udong,old,250,'old')
count_facil(Udong,child,250,'child')
count_facil(Udong,crosswork,250,'crosswork')
count_facil(Udong,accident,250,'accident')
Udong.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/Udong.csv')
#-----------------모델링----------------------------
X_data= Udong.drop('accident',axis=1)
y_data= Udong['accident']
alphas = [0,0.1,1,10,100]

'''alphas list 값을 반복하면서 alphas에 따른 평균 rmse를 구함.'''
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    
    # corss_val_score를 이용해 3폴드의 평균 RMSE를 계산
    neg_mse_scores = cross_val_score(ridge,X_data,y_data,scoring='neg_mean_squared_error',cv=5)
    avg_rmse= np.mean(np.sqrt(-1*neg_mse_scores))
    print('alpha{0} 일 때 5 folds의 평균 RMSE:{1:3f}'.format(alpha,avg_rmse))
    
'''
alpha0 일 때 5 folds의 평균 RMSE:1.048070
alpha0.1 일 때 5 folds의 평균 RMSE:1.048069
alpha1 일 때 5 folds의 평균 RMSE:1.048069
alpha10 일 때 5 folds의 평균 RMSE:1.048062
alpha100 일 때 5 folds의 평균 RMSE:1.048222
'''





