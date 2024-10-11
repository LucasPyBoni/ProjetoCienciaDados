#!/usr/bin/env python
# coding: utf-8

# # Projeto Airbnb Rio - Ferramenta de Previsão de Preço de Imóvel para pessoas comuns 

# ### Contexto
# 
# No Airbnb, qualquer pessoa que tenha um quarto ou um imóvel de qualquer tipo (apartamento, casa, chalé, pousada, etc.) pode ofertar o seu imóvel para ser alugado por diária.
# 
# Você cria o seu perfil de host (pessoa que disponibiliza um imóvel para aluguel por diária) e cria o anúncio do seu imóvel.
# 
# Nesse anúncio, o host deve descrever as características do imóvel da forma mais completa possível, de forma a ajudar os locadores/viajantes a escolherem o melhor imóvel para eles (e de forma a tornar o seu anúncio mais atrativo)
# 
# Existem dezenas de personalizações possíveis no seu anúncio, desde quantidade mínima de diária, preço, quantidade de quartos, até regras de cancelamento, taxa extra para hóspedes extras, exigência de verificação de identidade do locador, etc.
# 
# ### Nosso objetivo
# 
# Construir um modelo de previsão de preço que permita uma pessoa comum que possui um imóvel possa saber quanto deve cobrar pela diária do seu imóvel.
# 
# Ou ainda, para o locador comum, dado o imóvel que ele está buscando, ajudar a saber se aquele imóvel está com preço atrativo (abaixo da média para imóveis com as mesmas características) ou não.
# 
# ### O que temos disponível, inspirações e créditos
# 
# As bases de dados foram retiradas do site kaggle: https://www.kaggle.com/allanbruno/airbnb-rio-de-janeiro
# 
# Elas estão disponíveis para download abaixo da aula (se você puxar os dados direto do Kaggle pode ser que encontre resultados diferentes dos meus, afinal as bases de dados podem ter sido atualizadas).
# 
# Caso queira uma outra solução, podemos olhar como referência a solução do usuário Allan Bruno do kaggle no Notebook: https://www.kaggle.com/allanbruno/helping-regular-people-price-listings-on-airbnb
# 
# Você vai perceber semelhanças entre a solução que vamos desenvolver aqui e a dele, mas também algumas diferenças significativas no processo de construção do projeto.
# 
# - As bases de dados são os preços dos imóveis obtidos e suas respectivas características em cada mês.
# - Os preços são dados em reais (R$)
# - Temos bases de abril de 2018 a maio de 2020, com exceção de junho de 2018 que não possui base de dados
# 
# ### Expectativas Iniciais
# 
# - Acredito que a sazonalidade pode ser um fator importante, visto que meses como dezembro costumam ser bem caros no RJ
# - A localização do imóvel deve fazer muita diferença no preço, já que no Rio de Janeiro a localização pode mudar completamente as características do lugar (segurança, beleza natural, pontos turísticos)
# - Adicionais/Comodidades podem ter um impacto significativo, visto que temos muitos prédios e casas antigos no Rio de Janeiro
# 
# Vamos descobrir o quanto esses fatores impactam e se temos outros fatores não tão intuitivos que são extremamente importantes.

# ### Importar Bibliotecas e Bases de Dados

# In[77]:


from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split


# In[78]:


print(sns.__version__)


# In[79]:


caminho_dataset = Path('dataset')
meses = {'jan': 1, 'fev': 2, 'mar': 3, 'abr': 4, 'mai': 5, 'jun': 6, 'jul': 7, 'ago': 8, 'set': 9, 'out': 10, 'nov': 11, 'dez': 12}

bases = []

for arquivo in caminho_dataset.iterdir():
    nome_mes = arquivo.name[:3]
    mes_num = meses[nome_mes]
    ano = arquivo.name[-8:]
    ano = int(ano.replace('.csv',''))
    
    df = pd.read_csv(caminho_dataset / arquivo.name)
    df['mes'] = mes_num
    df['ano'] = ano
    bases.append(df)

df_base = pd.concat(bases) 
# display(df_base)

# tabela = pd.read_csv(caminho_dataset / r'abril2019.csv')
# display(tabela)


# ### Se tivermos muitas colunas, já vamos identificar quais colunas podemos excluir

# ###### para uma análise precisa e rápida, é necessário excluir colunas que não precisaremos
# 
# ##### tipos de coluna:
#     - coluna com linha vazias
#     - colunas com infos irrelevantes como foto do host
#     - colunas repetidas
#     - colunas com pouca informação
# 
# ##### pegando as 1000 primeiras linhas para avaliar

# In[80]:


# print(list(df_base.columns))
# df_base.head(1000).to_csv('primeiros registros.csv',sep=';')


# In[81]:


print(df_base['experiences_offered'].value_counts())


# In[82]:


print((df_base['host_listings_count']==df_base['host_total_listings_count']).value_counts())


# In[83]:


colunas = ['host_response_time','host_response_rate','host_is_superhost','host_listings_count','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','amenities','price','security_deposit','cleaning_fee','guests_included','extra_people','minimum_nights','maximum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','instant_bookable','is_business_travel_ready','cancellation_policy','mes','ano']

df_base = df_base.loc[:, colunas]


# In[84]:


# print(df_base['square_feet'].isnull().sum())


# ### Tratar Valores Faltando

# #### foram excluidas colunas onde número de linhas igual a NaN ultrapassaram 300.000 com drop
# #### as demais colunas que tinham poucas linhas com valores vazios foram limpadas com dropna

# In[85]:


for coluna in df_base:
    if df_base[coluna].isnull().sum() > 300000:
        df_base = df_base.drop(coluna, axis=1)

df_base = df_base.dropna()        


# ### Verificar Tipos de Dados em cada coluna

# ##### price e extra people estão como objeto, mas são valores

# In[86]:


# transformando price em float
df_base['price'] = df_base['price'].str.replace('$','')
df_base['price'] = df_base['price'].str.replace(',','')
df_base['price'] = df_base['price'].astype(np.float32, copy=False)

# transformando extra_people em float
df_base['extra_people'] = df_base['extra_people'].str.replace('$','')
df_base['extra_people'] = df_base['extra_people'].str.replace(',','')
df_base['extra_people'] = df_base['extra_people'].astype(np.float32, copy=False)

print(df_base.dtypes)


# ### Análise Exploratória e Tratar Outliers
# 
# - olhar feature(recurso) por feature
#     1. ver correlação entre feature pra manter ou não
#     2. Excluir outliers (usaremos como regra, valores abaixo de Q1 - 1.5xAmplitude e valores acima de Q3 + 1.5x Amplitude). Amplitude = Q3 - Q1
#     3. Confirmar se todas as features que temos fazem realmente sentido para o nosso modelo ou se alguma delas não vai nos ajudar e se devemos excluir
#     
# - Vamos começar pelas colunas de preço (resultado final que queremos) e de extra_people (também valor monetário). Esses são os valores numéricos contínuos.
# 
# - Depois vamos analisar as colunas de valores numéricos discretos (accomodates, bedrooms, guests_included, etc.)
# 
# - Por fim, vamos avaliar as colunas de texto e definir quais categorias fazem sentido mantermos ou não.
# 
# MAS CUIDADO: não saia excluindo direto outliers, pense exatamente no que você está fazendo. Se não tem um motivo claro para remover o outlier, talvez não seja necessário e pode ser prejudicial para a generalização. Então tem que ter uma balança ai. Claro que você sempre pode testar e ver qual dá o melhor resultado, mas fazer isso para todas as features vai dar muito trabalho.
# 
# Ex de análise: Se o objetivo é ajudar a precificar um imóvel que você está querendo disponibilizar, excluir outliers em host_listings_count pode fazer sentido. Agora, se você é uma empresa com uma série de propriedades e quer comparar com outras empresas do tipo também e se posicionar dessa forma, talvez excluir quem tem acima de 6 propriedades tire isso do seu modelo. Pense sempre no seu objetivo

# In[87]:


plt.figure(figsize=(12, 8))
sns.heatmap(df_base.corr(numeric_only=True), annot=True, cmap='Greens')
plt.show()


# ### Funções para análise de outliers

# In[88]:


def limites(coluna):
    q1 = coluna.quantile(0.25)
    q3 = coluna.quantile(0.75)
    amplitude = q3 - q1
    return q1 - 1.5 * amplitude, q3 + 1.5 * amplitude
def excluir_outliers(df, nome_coluna):
    lim_inf, lim_sup = limites(df[nome_coluna])
    qtd_linhas = df.shape[0]
    df = df.loc[(df[nome_coluna] >= lim_inf) & (df[nome_coluna] <= lim_sup) , :]
    linhas_removidas = qtd_linhas - df.shape[0]
    return df, linhas_removidas


# In[89]:


def diagrama(coluna):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)
    sns.boxplot(x=coluna, ax=ax1)
    ax2.set_xlim(limites(coluna))
    sns.boxplot(x=coluna, ax=ax2)
    
def histograma(coluna):
    plt.figure(figsize=(12,4))
    sns.histplot(coluna, element='bars')
    
def grafico_barra(coluna):
    plt.figure(figsize=(12,4))
    ax = sns.barplot(x=coluna.value_counts().index, y=coluna.value_counts())
    ax.set_xlim(limites(coluna))


# ### Price

# In[90]:


diagrama(df_base['price'])
histograma(df_base['price'])


# In[91]:


df_base, linhas_removidas = excluir_outliers(df_base, 'price')
print(f'{linhas_removidas} linhas removidas')


# In[92]:


histograma(df_base['price'])
print(df_base.shape)


# ### extra people

# In[93]:


histograma(df_base['extra_people'])
diagrama(df_base['extra_people'])


# In[94]:


df_base, linhas_removidas = excluir_outliers(df_base, 'extra_people')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
histograma(df_base['extra_people'])


# ### host_listings_count

# In[95]:


diagrama(df_base['host_listings_count'])
grafico_barra(df_base['host_listings_count'])


# In[96]:


df_base, linhas_removidas = excluir_outliers(df_base, 'host_listings_count')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['host_listings_count'])


# ### accommodates

# In[97]:


diagrama(df_base['accommodates'])
grafico_barra(df_base['accommodates'])


# In[98]:


df_base, linhas_removidas = excluir_outliers(df_base, 'accommodates')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['accommodates'])


# ### bathrooms

# In[99]:


diagrama(df_base['bathrooms'])
grafico_barra(df_base['bathrooms'])
plt.figure(figsize=(12,4))
sns.barplot(x=df_base['bathrooms'].value_counts().index, y=df_base['bathrooms'].value_counts())


# In[100]:


df_base, linhas_removidas = excluir_outliers(df_base, 'bathrooms')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['bathrooms'])


# ### bedrooms

# In[101]:


diagrama(df_base['bedrooms'])
grafico_barra(df_base['bedrooms'])


# In[102]:


df_base, linhas_removidas = excluir_outliers(df_base, 'bedrooms')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['bedrooms'])



# ### beds

# In[103]:


diagrama(df_base['beds'])
grafico_barra(df_base['beds'])


# In[104]:


df_base, linhas_removidas = excluir_outliers(df_base, 'beds')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['beds'])


# ### guests_included

# In[105]:


# diagrama(df_base['guests_included'])
# grafico_barra(df_base['guests_included'])
print(limites(df_base['guests_included']))
plt.figure(figsize=(15, 5))
sns.barplot(x=df_base['guests_included'].value_counts().index, y=df_base['guests_included'].value_counts())


# ##### conforme gráfico acima, guests included se mostrou uma feature ruim, pois provavelmente os clientes por padrão colocam 1, ou seja, essa coluna será excluida

# In[106]:


df_base = df_base.drop('guests_included', axis=1)
df_base.shape


# ### minimum_nights

# In[107]:


diagrama(df_base['minimum_nights'])
grafico_barra(df_base['minimum_nights'])


# In[108]:


df_base, linhas_removidas = excluir_outliers(df_base, 'minimum_nights')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['minimum_nights'])       
          


# ### maximum_nights

# In[109]:


diagrama(df_base['maximum_nights'])
grafico_barra(df_base['maximum_nights'])


# In[110]:


df_base = df_base.drop('maximum_nights', axis=1)
df_base.shape


# ### number_of_reviews   

# In[111]:


diagrama(df_base['number_of_reviews'])
grafico_barra(df_base['number_of_reviews'])


# ##### se tirar os outliers da coluna 'number_of_reviews' os hosts que possuem acima de 15 reviews serão descartados por conta dos limites, impactando negativamente. Todavia, ao deixar a coluna, os hosts que ainda não receberam avaliações serão impactados igualmente. Dessa forma, vamos excluir essa coluna da análise

# In[112]:


df_base = df_base.drop('number_of_reviews', axis=1)
df_base.shape


# ### TRATANDO COLUNAS DE TEXTO

# - property_type

# In[113]:


print(df_base['property_type'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='property_type')
grafico.tick_params(axis='x', rotation=90)


# - ao invés de excluir os pequenos value counts de uma coluna de texto, é mais viável agrupa-los
# - agruparemos todos que tem abaixo de 2000 e soma-los a coluna outros

# In[114]:


lista_aux = df_base['property_type'].value_counts()
itens_abaixo_5000 = []
for item in lista_aux.index:
    if lista_aux[item] < 5000:
        itens_abaixo_5000.append(item)
for tipo in itens_abaixo_5000:
    df_base.loc[df_base['property_type'] == tipo, 'property_type'] = 'outros'
        
        
print(df_base['property_type'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='property_type')
grafico.tick_params(axis='x', rotation=90)


# - room_type

# In[115]:


print(df_base['room_type'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='room_type')
grafico.tick_params(axis='x', rotation=90)


# - bed_type

# In[116]:


print(df_base['bed_type'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='bed_type')
grafico.tick_params(axis='x', rotation=90)

lista_aux = df_base['bed_type'].value_counts()
itens_abaixo = []

for item in lista_aux.index:
    if lista_aux[item] < 10000:
        itens_abaixo.append(item)
        
for tipo in itens_abaixo:
    df_base.loc[df_base['bed_type']==tipo, 'bed_type'] = "Outros"
    
print(df_base['bed_type'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='bed_type')
grafico.tick_params(axis='x', rotation=90)


# - cancellation_policy

# In[117]:


print(df_base['cancellation_policy'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='cancellation_policy')
grafico.tick_params(axis='x', rotation=90)

lista_aux = df_base['cancellation_policy'].value_counts()
itens_abaixo = []

for item in lista_aux.index:
    if lista_aux[item] < 10000:
        itens_abaixo.append(item)
        
for tipo in itens_abaixo:
    df_base.loc[df_base['cancellation_policy']==tipo, 'cancellation_policy'] = "Strict Normal"
    
print(df_base['cancellation_policy'].value_counts())
plt.figure(figsize=(12, 4))
grafico = sns.countplot(data=df_base, x='cancellation_policy')
grafico.tick_params(axis='x', rotation=90)


# - amenities
# 
# são muitos dados a analisar, por isso optamos por analisar a quantidade amenities, e não cada um individualmente

# In[118]:


print(df_base['amenities'].iloc[1].split(','))
print(len(df_base['amenities'].iloc[1].split(',')))

df_base['num_amenities'] = df_base['amenities'].str.split(',').apply(len)
df_base = df_base.drop('amenities', axis=1)


# In[119]:


df_base, linhas_removidas = excluir_outliers(df_base, 'num_amenities')
print(f'{linhas_removidas} linhas removidas')
print(df_base.shape)
grafico_barra(df_base['num_amenities'])       


# ### Visualização de Mapa das Propriedades
# 
# Vamos criar um mapa que exibe um pedaço da nossa base de dados aleatório (50.000 propriedades) para ver como as propriedades estão distribuídas pela cidade e também identificar os locais de maior preço 

# In[120]:


amostra = df_base.sample(n=50000)
centro_mapa = {'lat':amostra.latitude.mean(), 'lon':amostra.longitude.mean()}
mapa = px.density_mapbox(amostra, lat='latitude', lon='longitude',z='price', radius=2.5,
                        center=centro_mapa, zoom=10,
                        mapbox_style='stamen-terrain')
mapa.show()


# ### Encoding

# - features(recursos) de valores True ou False, serão substituidos por 1 = True e 0 = False
# - features de categoria usaremos variaveis dummies

# In[121]:


df_base_cod = df_base.copy()
colunas_df = ['host_is_superhost','instant_bookable','is_business_travel_ready']
for coluna in colunas_df:
    df_base_cod.loc[df_base_cod[coluna]=='f', coluna] = 0
    df_base_cod.loc[df_base_cod[coluna]=='t', coluna] = 1
    
print(df_base_cod.iloc[500])


# In[122]:


colunas_cat = ['bed_type','cancellation_policy','room_type','property_type']
df_base_cod = pd.get_dummies(data=df_base_cod, columns=colunas_cat, dtype=int)
display(df_base_cod.head(10))


# ### Modelo de Previsão

# - metricas de avaliação
# 
# r² > quanto mais próximo de 1 ou 100% melhor... vai dizer o quão bom é modelo
# 
# Erro quadrático médio > quanto menor, melhor... vai dizer qual o modelo erra menos e por quanto erra

# In[123]:


def avaliar_modelo(nome_modelo, y_teste, previsao):
    r2 = r2_score(y_teste, previsao)
    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
    return f'Modelo {nome_modelo}\nR²: {r2:.2%}\nRSME: {RSME:.2f}'


# - Escolha dos Modelos a Serem Testados
#     1. RandomForest
#     2. LinearRegression
#     3. Extra Tree
#     
# Esses são alguns dos modelos que existem para fazer previsão de valores numéricos (o que chamamos de regressão). Estamos querendo calcular o preço, portanto, queremos prever um valor numérico.
# 
# Assim, escolhemos esses 3 modelos. Existem dezenas, ou até centenas de modelos diferentes. A medida com que você for aprendendo mais e mais sobre Ciência de Dados, você vai aprender sempre novos modelos e entendendo aos poucos qual o melhor modelo para usar em cada situação.
# 
# Mas na dúvida, esses 3 modelos que usamos aqui são bem bons para muitos problemas de Regressão.

# In[124]:


modelo_rf = RandomForestRegressor()
modelo_lr = LinearRegression()
modelo_et = ExtraTreesRegressor()

modelos = {
    'RandomForest': modelo_rf,
    'LinearRegression': modelo_lr,
    'ExtraTrees': modelo_et
          }


y = df_base_cod['price']
x = df_base_cod.drop('price',axis=1)


# - Separa os dados em treino e teste + Treino do Modelo
# 
# Essa etapa é crucial. As Inteligências Artificiais aprendendo com o que chamamos de treino.
# 
# Basicamente o que a gente faz é: a gente separa as informações em treino e teste, ex: 10% da base de dados vai servir para teste e 90% para treino (normalmente treino é maior mesmo)
# 
# Aí, damos para o modelo os dados de treino, ele vai olhar aqueles dados e aprender a prever os preços.
# 
# Depois que ele aprende, você faz um teste com ele, com os dados de teste, para ver se ela está bom ou não. Analisando os dados de teste você descobre o melhor modelo

# In[125]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

for nome_modelo, modelo in modelos.items():
    # treinar
    modelo.fit(x_train, y_train)
    # teste
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# ### Análise do Melhor Modelo

# In[126]:


for nome_modelo, modelo in modelos.items():
    # teste
    previsao = modelo.predict(x_test)
    print(avaliar_modelo(nome_modelo, y_test, previsao))


# - o melhor modelo foi o ExtraTreesRegressor, pois apresentou R² mais alto e menor erro quadrático médio, além disso, a velocidade de execução não foi tão diferente
# 
# O pior foi o LinearRegression
# 
# - Resultados do modelo vencedor (ExtraTreesRegressor):
# Modelo ExtraTrees
# R²: 97.50%
# RSME: 41.91

# ### Ajustes e Melhorias no Melhor Modelo

# In[127]:


# print(modelo_et.feature_importances_)
# print(x_train.columns)

importancia_feature = pd.DataFrame(modelo_et.feature_importances_, x_train.columns)
importancia_feature = importancia_feature.sort_values(by = 0, ascending = False)
plt.figure(figsize=(12, 4))
ax = sns.barplot(x=importancia_feature.index, y=importancia_feature[0])
ax.tick_params(axis='x', rotation=90)


# - coluna 'is_business_travel_ready' estava sendo irrelevante para nossa análise, por isso ficamos com a decisão de excluir ela da base de dados a fim de melhorar nosso modelo

# In[128]:


df_base_cod = df_base_cod.drop('is_business_travel_ready', axis=1)

y = df_base_cod['price']
x = df_base_cod.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(x_train, y_train)
# teste
previsao = modelo_et.predict(x_test)
print(avaliar_modelo("ExtraTrees", y_test, previsao))


# In[129]:


df_base_cod_teste = df_base_cod.copy()
for coluna in df_base_cod_teste:
    if 'bed_type' in coluna:
        df_base_cod_teste = df_base_cod_teste.drop(coluna, axis=1)
        
y = df_base_cod_teste['price']
x = df_base_cod_teste.drop('price', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)

modelo_et.fit(x_train, y_train)
# teste
previsao = modelo_et.predict(x_test)
print(avaliar_modelo("ExtraTrees", y_test, previsao))


# In[130]:


print(len(previsao))


# # Deploy do Projeto
# 
# - Passo 1 -> Criar arquivo do Modelo (joblib)<br>
# - Passo 2 -> Escolher a forma de deploy:
#     - Arquivo Executável + Tkinter
#     - Deploy em Microsite (Flask)
#     - Deploy apenas para uso direto Streamlit
# - Passo 3 -> Outro arquivo Python (pode ser Jupyter ou PyCharm)
# - Passo 4 -> Importar streamlit e criar código
# - Passo 5 -> Atribuir ao botão o carregamento do modelo
# - Passo 6 -> Deploy feito

# In[132]:


x['price'] = y
x.to_csv('dados_analise.csv')


# In[133]:


import joblib
joblib.dump(modelo_et,'modelo.joblib')


# In[ ]:




