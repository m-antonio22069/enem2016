# enem2016
## Utilização de algoritmos de regressão para estimar as notas de matemática no Enem 2016.

### Introdução
Muitas universidades brasileiras utilizam o ENEM para selecionar seus futuros alunos e alunas. 
Isto é feito com uma média ponderada das notas das provas de matemática, ciências da natureza,
linguagens e códigos, ciências humanas e redação, com os pesos abaixo:

- matemática: 3
- ciências da natureza: 2
- linguagens e códigos: 1.5
- ciências humanas: 1
- redação: 3

### Solução

**A primeira parte do projeto consiste em importar as bibliotecas necessarias:**

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('ggplot')
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
      
**Apos isso, precisaremos carregar a base de dados:**

    dataset_train = pd.read_csv('train.csv')
    dataset_test = pd.read_csv('test.csv')

    dados = pd.DataFrame()
    index = 0
    for i in dataset_test.columns:
       dados.insert(loc=index, column=i, value=dataset_train[i])
       index+=1

    dados.insert(loc=47, column='NU_NOTA_MT', value=dataset_train['NU_NOTA_MT'])
    
**Pode-se observar que os dados nao se apresentam de maneira agradavel,
portanto, para que seja extraida as melhores Features, é preciso fazer um tratamento antes.**

**Tratando dados nao essenciais ou com muitas faltas de informação:**
    
    dados = dados.drop(columns=['NU_INSCRICAO'])
    dados = dados.drop(columns = ['SG_UF_RESIDENCIA'])
    dados = dados.drop(columns=['TP_DEPENDENCIA_ADM_ESC'])
    dados = dados.drop(columns = ['CO_PROVA_CN'])
    dados = dados.drop(columns = ['CO_PROVA_CH'])
    dados = dados.drop(columns = ['CO_PROVA_LC'])
    dados = dados.drop(columns = ['CO_PROVA_MT'])
    dados = dados.drop(columns = ['Q027'])
    dados = dados.drop(columns = ['TP_ENSINO'])
    
    
**Tratando dados nulos:** 

    dados.isnull().sum()
    dados['NU_NOTA_CN']=dados['NU_NOTA_CN'].fillna(0)
    dados['NU_NOTA_CH']=dados['NU_NOTA_CH'].fillna(0)
    dados['NU_NOTA_LC']=dados['NU_NOTA_LC'].fillna(0)
    dados['NU_NOTA_MT']=dados['NU_NOTA_MT'].fillna(0)

    dados['TP_STATUS_REDACAO']=dados['TP_STATUS_REDACAO'].fillna(2)

    dados['NU_NOTA_COMP1']=dados['NU_NOTA_COMP1'].fillna(0)
    dados['NU_NOTA_COMP2']=dados['NU_NOTA_COMP2'].fillna(0)
    dados['NU_NOTA_COMP3']=dados['NU_NOTA_COMP3'].fillna(0)
    dados['NU_NOTA_COMP4']=dados['NU_NOTA_COMP4'].fillna(0)
    dados['NU_NOTA_COMP5']=dados['NU_NOTA_COMP5'].fillna(0)

    dados['NU_NOTA_REDACAO']=dados['NU_NOTA_REDACAO'].fillna(0)
    
**Codificando dados String => Numerico:**

    dados['TP_SEXO']=dados['TP_SEXO'].replace('F',0)
    dados['TP_SEXO']=dados['TP_SEXO'].replace('M',1)
 
    from collections import Counter

    questoes = ['Q001','Q002','Q006', 'Q024','Q025','Q026', 'Q047']

    for questao in questoes:
       counter = sorted(Counter(dados[questao]))
       for letra in counter:
          dados[questao]=dados[questao].replace(letra, counter.index(letra))
          
**Apos feita a limpeza dos dados, é necessario extrair as melhores features para 
fornecer ao modelo. Dessa forma, para que fique mais claro a visualização, 
foi plotado um grafico que mostra o grau de correlação de cada coluna com 
a nota final de matematica.**

**Extraindo correlação entre colunas:**

    correlacao = dados.corrwith(dados['NU_NOTA_MT']).iloc[:-1].to_frame()
    correlacao['abs']=correlacao[0].abs()
    correlacao_ordenada = correlacao.sort_values('abs',ascending=False)[0]

    fig, ax = plt.subplots(figsize=(10,20))
    sns.heatmap(correlacao_ordenada.to_frame(),cmap='coolwarm',annot=True, 
            vmin=1, vmax=1, ax=ax)

**Foi obtida a seguinte Imagem:**
![Figure_1](https://user-images.githubusercontent.com/56478707/83073591-14c81300-a047-11ea-9d79-e9768ec9c524.png)

**Pode-se observar que determinadas Features tem mais correlação com a nota de matematica do que outras, logo, 
vamos extrair do dataset estas features e analisa-las**

**Removendo Features com baixa correlação:**

        dict(correlacao_ordenada)
        key = list(correlacao_ordenada.keys())

        corr = correlacao_ordenada[key.index('Q001'):]

        for colunm in corr.keys():
           dados = dados.drop(columns = colunm)
           
**Separando classe e variavel e dividindo a base de dados:**

        variaveis = dados.iloc[:,:21].values
        classe = dados.iloc[:,21].values

        from sklearn.model_selection import train_test_split

        x_treino, x_teste, y_treino, y_teste = train_test_split(variaveis,
                                                                classe,
                                                                test_size=0.3,
                                                                random_state=0)
       
       
**Criando piperlines para otimização dos testes. Dessa forma, vamos testar 2 modelos de regressão e verificar qual deles
tem maior taxa de acerto. Alem disso, vamos verifiar como a padronização e a normalização dos dados influencia no resultado**

        from sklearn.pipeline import Pipeline

        pip_std_ForestRandom = Pipeline([('scaler',StandardScaler()),
                                 ('RandomForestRegressor', RandomForestRegressor())])

        pip_min_max_ForestRandom = Pipeline([('minmax',MinMaxScaler()),
                                 ('RandomForestRegressor', RandomForestRegressor())])

        pip_std_LinearRegression = Pipeline([('scaler',StandardScaler()),
                                 ('RandomForestRegressor', LinearRegression())])

        pip_min_max_LinearRegression = Pipeline([('minmax',MinMaxScaler()),
                                 ('RandomForestRegressor', LinearRegression())])
                                 
 - **Random Forest**
 
**treinando modelo Random Forest std e min_max, alem de retirar seus respectivos scores**

        pip_std_ForestRandom.fit(x_treino,y_treino)
        pip_std_ForestRandom.score(x_teste, y_teste)
        >> 0.919802716056002
        
        pip_min_max_ForestRandom.fit(x_treino, y_treino)
        pip_min_max_ForestRandom.score(x_teste, y_teste)
        >> 0.9190858943441254
        
 - **Linear Regressor**
 
**treinando modelo Linear Regression std e min_max, alem de retirar seus respectivos scores**

        pip_std_LinearRegression.fit(x_treino, y_treino)
        pip_std_LinearRegression.score(x_teste,y_teste)
        >> 0.9100480312060392
        
        pip_min_max_LinearRegression.fit(x_treino, y_treino)
        pip_min_max_LinearRegression.score(x_teste,y_teste)
        >> 0.9100498015669882
        
        
### Conclusão

Pode-se observar que não houve muita diferença na taxa de acerto, porem, este problema foi importante
para botar em pratica algumas tecnicas de pre-processamento e analise de dados. Cabe destacar a importancia 
de analisar e filtrar os dados previamente, haja visto que os algoritmos são altamente sensiveis a qualidade
dos atributos. Dessa forma, se a seleção for ineficiente, o modelo de Machine Learning tende a não te o 
desempenho esperado .Neste problema, foi possivel obter modelos com um score acima de 90%.
