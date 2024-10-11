#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import streamlit as st
import joblib


# modelo = joblib.load('modelo.joblib')

x_num = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 'extra_people': 0,
'minimum_nights': 0, 'ano': 0, 'mes': 0, 'num_amenities': 0, 'host_listings_count': 0}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_listas = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'House', 'Loft', 'outros', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'cancellation_policy': ['flexible', 'moderate', 'Strict Normal', 'strict_14_with_grace_period']
            }


dic = {}

for item in x_listas:
    for lista in x_listas[item]:
        dic[f'{item}_{lista}'] = 0

for item in x_num:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format='%.5f')
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}',step=1, value=0)
    x_num[item] = valor
        
    
for item in x_tf:
    valor = st.selectbox(f'{item}',('sim', 'não'))
    if valor == 'sim':
        x_tf[item] = 1
    else:
        x_tf[item] = 0
    
for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dic[f'{item}_{valor}'] = 1 
botao = st.button('Previsão do valor do Imóvel ')

if botao:
    dic.update(x_num)
    dic.update(x_tf)
    valores_x = pd.DataFrame(dic, index=[0])
    tabela = pd.read_csv('dados_analise.csv')
    colunas = list(tabela.columns)[1:-1] 
    valores_x = valores_x[colunas]
    
    modelo = joblib.load('modelo.joblib')
    preco = modelo.predict(valores_x)
    st.write(preco)


# In[ ]:




