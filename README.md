## Resumo do projeto

No Solução Airbnb tem o projeto e todas explicações de tudo, mas resumidamente o pandas lê a tabela, todas os tratamentos como retirar colunas inúteis, análise de colunas que podem atrapalhar, exclusão de outliers e as informações são transformadas em número 1 pra True e 0 False para que a IA possa realizar seus cálculos. A IA escolhida foi a ExtraTrees devido ao baixo Erro Quadrático e o alto R², após seu treinamento e teste de previsão ela foi salvada por meio do Joblib para o deploy.

## Deploy do projeto

São usadas 3 bibliotecas: pandas para ler o arquivo, streamlit para realizar o deploy em nova aba, já o joblib carrega o modelo de IA salvo e treinado.

No deploy cria-se três dicionários, x_num contendo números normais, x_tf contendo 1 pra True e 0 para False e x_listas que contém escolhas para o usuário fazer, que depois são transformados em número também.
Cada chave do dicionário é um input para o usuário preencher ou selecionar.
perto do final cria-se o botão, se o botão for selecionado, o dicionário é atualizado (update), um novo cálculo é feito, e o valor é exibido no final.

