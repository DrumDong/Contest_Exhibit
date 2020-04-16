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
old_facil2 = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/노인보호구역.csv',encoding='cp949')
dolbom = pd.read_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/돌봄시설.csv',encoding='cp949')

#old_class
old_class=old_class[['소재지전체주소','위치정보(X)','위치정보(Y)']]
old_class = old_class[old_class['소재지전체주소'].str.contains('동작')]

#old_facil
old_facil = old_facil[['소재지전체주소','위치정보(X)','위치정보(Y)']]
old_facil = old_facil[old_facil['소재지전체주소'].str.contains('동작')]

#old_facil2
old_facil2 = old_facil2[['소재지지번주소','위도','경도']]
old_facil2 = old_facil2[['소재지지번주소','경도','위도']]
old_facil2.columns=['소재지전체주소','위치정보(X)','위치정보(Y)']

#dolbom
dolbom = dolbom[['지번주소','좌표정보','좌표정보.1']]
dolbom.columns=['소재지전체주소','위치정보(X)','위치정보(Y)']

===='좌표변환'===
old_class[['new_x','new_y']]=old_class.apply(transform_w84_to_utmk, axis=1) # epsg:2097
old_facil[['new_x','new_y']]=old_facil.apply(transform_w84_to_utmk, axis=1) # epsg:2097
old_facil2[['new_x','new_y']]=old_facil2.apply(transform_w84_to_utmk, axis=1) # epsg:2097
dolbom[['new_x','new_y']]=dolbom.apply(transform_w84_to_utmk, axis=1) # epsg:4326

old_place=pd.concat([old_class,old_facil,dolbom])
old_place = pd.concat([old_place,old_facil2])
old_place.to_csv('C:/Users/ehdrb/Desktop/동작구 빅데이터 공모전 데이터/old_place.csv')