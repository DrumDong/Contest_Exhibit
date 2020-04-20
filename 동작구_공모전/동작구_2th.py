# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:42:46 2020

@author: ehdrb
"""

import pandas as pd
from pyproj import Proj,transform


proj_WGS84 = Proj(init='epsg:4326') # epsg:4326
proj_UTMK = Proj(init='epsg:5186')
def transform_w84_to_utmk(df):
    return pd.Series(transform(proj_WGS84,proj_UTMK,df['위치정보(X)'],df['위치정보(Y)']),index=['위치정보(X)','위치정보(Y)'])

---'''동작구 노인시설'''---
old_class= pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/서울시노인교실.csv',encoding='cp949')
old_facil = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/서울시노인요양시설.csv',encoding='cp949')
#old_facil2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/노인보호구역.csv',encoding='cp949')
dolbom = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/돌봄시설.csv',encoding='cp949')

#old_class
old_class=old_class[['소재지전체주소','위치정보(X)','위치정보(Y)']]
old_class = old_class[old_class['소재지전체주소'].str.contains('동작')]

#old_facil
old_facil = old_facil[['소재지전체주소','위치정보(X)','위치정보(Y)']]
old_facil = old_facil[old_facil['소재지전체주소'].str.contains('동작')]

#old_facil2
#old_facil2 = old_facil2[['소재지지번주소','위도','경도']]
#old_facil2 = old_facil2[['소재지지번주소','경도','위도']]
#old_facil2.columns=['소재지전체주소','위치정보(X)','위치정보(Y)']

#dolbom
dolbom = dolbom[['지번주소','좌표정보','좌표정보.1']]
dolbom.columns=['소재지전체주소','위치정보(X)','위치정보(Y)']

===='좌표변환'===
old_class[['new_x','new_y']]=old_class.apply(transform_w84_to_utmk, axis=1) # epsg:2097
old_facil[['new_x','new_y']]=old_facil.apply(transform_w84_to_utmk, axis=1) # epsg:2097
#old_facil2[['new_x','new_y']]=old_facil2.apply(transform_w84_to_utmk, axis=1) # epsg:2097
dolbom[['new_x','new_y']]=dolbom.apply(transform_w84_to_utmk, axis=1) # epsg:4326

old_place=pd.concat([old_class,old_facil,dolbom])
#old_place = pd.concat([old_place,old_facil2])
old_place.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/old_place.csv')

---'''동작구 어린이시설'''---
child1 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/동작구_어린이집_1(UTM_K).csv',encoding='cp949')
child2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/동작구_유치원_1.csv(UTM_K).csv',encoding='cp949')
child3 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/전국학교.csv',encoding='cp949')
child1=child1[['소재지도로명주소','lat','lon','UTM-K_경도변환','UTM-K_위도변환']]
child2=child2[['주소','lat','lon','UTM-K_경도변환','UTM-K_위도변환']]
child1.columns = ['주소','위도','경도','new_x','new_y']
child2.columns = ['주소','위도','경도','new_x','new_y']

child3=child3[['학교명','학교급구분','소재지지번주소','위도','경도']]
child3=child3[(child3['학교급구분']=='초등학교')&(child3['소재지지번주소'].str.contains('동작구'))]
child3.rename(columns={'위도':'위치정보(Y)',
                       '경도':'위치정보(X)'},inplace=True)
child3[['new_x','new_y']] = child3.apply(transform_w84_to_utmk,axis=1)
child3= child3[['소재지지번주소','위치정보(Y)','위치정보(X)','new_x','new_y']]
child3.columns = ['주소','위도','경도','new_x','new_y']

child = pd.concat([child1,child2,child3])
child.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/child_place.csv')


    
    
    
    