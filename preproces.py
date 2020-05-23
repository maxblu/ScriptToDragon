import streamlit as st
import pandas as pd
import json
from os import listdir
import numpy as np
from tqdm import tqdm
# data = pd.read_csv('train.csv',usecols=['title', 'text'],encoding='utf-8' )
# count= 0
# for i in range(0,330000,30000):
#     aux = data[i:i+30000]
#     aux.to_csv("part"+str(count)+'.csv' )
#     count+=1

root = 'news_data/'

jsons = listdir(root)

df = pd.DataFrame(columns=['Title','Text'])

for noti in tqdm(jsons):
    with open( root+noti,'r', encoding='utf-8') as fd: 
        js = json.load(fd)
        text = js['text']
        title = js['title']
        aux = pd.DataFrame([[title,text]],columns=['Title','Text'])
        df = df.append(aux)

df.to_csv('train.csv',encoding='utf-8')

# df

# data = pd.read_csv('traintest.csv',usecols=['Title', 'Text'],encoding='utf-8' )
# data