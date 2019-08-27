---
author:
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
- 
bibliography:
- 'refs.bib'
title: |
    Explora��o e an�lise de datasets e t�cnicas de aumento de dados para
    identifica��o de idioma (LID) utilizando redes neurais artificiais
---

\maketitle
abstract
========

Artificial Neural Networks (ANNs) are \"black box\" machine learning
algorithms, which means that the internal process of the developed
system or model can not be easily understood. This implies that the data
represents an important role for the trainning process of an ANN. In
special, the data selection and the data pre-processing process is a
decisive factor and directly impacts the algorithm performance. This
work presents an extensive reasearch and exploration of different speech
audio bases for Spoken Language Identification (LID) using an ANN model,
and analyses the impact of different preprocessing techniques and data
augmentation on the model performance.

resumo
======

Redes Neurais Artificiais (RNAs) s�o algoritmos de aprendizado de
m�quina \"caixa preta\", o que significa que o processo interno do
modelo ou sistema desenvolvido n�o � facilmente entendido, isso implica
que os dados representam um papel muito importante no treinamento de uma
RNA. Em especial, a escolha dos dados de entrada para o treinamento dos
modelos e o pr�-processamento dos dados desempenham fator decisivo e
influenciam diretamente o desempenho do algoritmo. Neste trabalho, s�o
realizadas explora��es de diferentes bases de �udios de discursos para a
identifica��o de idioma (LID) utilizando um modelo de RNA, analisando o
impacto de diferentes t�cnicas de pr�-processamento e aumento de dados
no desempenho do modelo.

Introdu��o {#sec:intro}
==========

De uma forma geral, Redes Neurais Artificiais s�o m�quinas projetadas
para *modelarem* a forma como o c�rebro humano realiza uma tarefa
particular ou alguma fun��o de interesse [@haykin2007redes]. De fato, o
uso de modelos de RNAs t�m se tornado cada vez mais frequente em
diferentes campos de pesquisa, e se mostraram extremamente eficazes nas
mais diversas tarefas como reconhecimento de padr�es, classifica��o de
imagens, vis�o computacional, transcri��o de �udio, entre outras. Uma
etapa essencial no processo de desenvolvimento de sistemas inteligentes
� a sele��o e o pr�-processamento de dados a serem utilizados no
processo de aprendizado de m�quina, e com as RNAs, esse processo pode
ser ainda mais importante. Neste trabalho, s�o explorados e analisados
diferentes t�cnicas quanto ao desenvolvimento de um dataset adequado ao
treinamento de um modelo de RNA para a identifica��o de idioma.

Uma das �reas de interesse das RNAs s�o aplica��es em �udio. As
pesquisas na �rea tiveram in�cio em 1952 a partir de um sistema
dependente de locutor que era capaz de reconhecer d�gitos falados
[@juang2005automatic]. Pesquisas recentes na �rea de reconhecimento de
idioma incluem [@revay2019multiclass], [@bartz2017language],
[@richardson2015deep], [@zazo2016language] e [@montavon2009deep]. Nestes
trabalhos, diferentes abordagens foram utilizadas quanto a sele��o de
bases de �udios, pr�-processamento e extra��o de caracter�sticas.

No caso da identifica��o de idioma existem diversas bases de �udios que
podem ser utilizadas, como por exemplo, leituras de �udio livros ou
bases de discursos preparadas para tarefas relacionadas a reconhecimento
de discursos. Cada base apresenta seus fatores positivos e negativos.
Por exemplo, a leitura de um �udio livro �, em geral, feita apenas por
um locutor, o que implica em uma quantidade pobre de diferentes timbres
de vozes nos dados. Em contraste, uma base de discursos como a
LibriSpeech [@panayotov2015librispeech] ou a VoxForge [^1] podem
apresentar muito ru�do ou certas caracter�sticas indesejadas. Al�m
disso, para o sucesso do aprendizado do modelo, � importante garantir
uma consist�ncia nos dados entre as classes (no caso, os idiomas), caso
contr�rio o modelo poder� aprender conceitos desinteressantes para o
prop�sito do problema, o que gerar� um modelo com pouca capacidade de
classificar corretamente novas inst�ncias.

A tarefa de pr�-processamento dos �udios � fundamental, principalmente
porque certas caracter�sticas do �udio dependem da origem destes e do
pr�-processamento j� previamente realizado nestes dados, al�m de
convers�es entre formatos, qualidade de grava��o e presen�a ou n�o de
ru�dos no momento da capta��o das vozes. No caso da utiliza��o de bases
de discursos o pr�-processamento pode desempenhar tarefa ainda mais
importante, principalmente porque cada base de discurso apresenta
caracter�sticas intr�nsecas que devem ser verificadas e, se necess�rio,
corrigidas de alguma forma. Por exemplo, uma das bases utilizadas no
desenvolvimento dos dados de treinamento prov�m de grava��es de r�dio no
idioma espanhol [@hernandez2014ciempiess].

Outra tarefa importante no desenvolvimento de dados para a utiliza��o em
modelos de aprendizado de m�quina � a realiza��o de t�cnicas de aumento
de dados. O aumento de dados consiste na gera��o de novas inst�ncias a
partir de dados antigos utilizando t�cnicas espec�ficas. Em geral, o
aumento de dados promove um melhor desempenho no aprendizado do modelo
porque uma quantidade maior de dados promove uma melhor capacidade de
generaliza��o e de aprendizado de caracter�sticas importantes. No mundo
do reconhecimento de imagens, uma das t�cnicas para aumento de dados � o
espelhamento, a rota��o e a distor��o. No caso dos �udios, podem ser
utilizadas t�cnicas como adi��o de ru�do de fundo e mudan�a de entona��o
e velocidade.

O objetivo geral deste artigo � analisar extensivamente alguns dos
poss�veis conjuntos de dados a serem utilizados por um modelo de RNA a
fim de identificar com �xito a l�ngua falada, partindo da sele��o dos
dados at� t�cnicas de pr�-processamento e aumento de dados.

Espera-se avaliar o impacto dos dados no desenvolvimento de um modelo
RNA para identifica��o de idioma:

1.  Quanto a sele��o dos dados, isto �, �udio livros ou bases de
    discursos nos idiomas alvo.

2.  Quanto as t�cnicas de pr�-processamento de �udio, ou seja, a taxa de
    amostragem utilizada, a aplica��o de filtro passa-baixa, a remo��o
    de per�odos longos de sil�ncio e a normaliza��o de �udio.

3.  Quanto ao uso de t�cnicas de aumento de dados, no caso, a mudan�a de
    velocidade, a adi��o de ru�do e/ou a mudan�a de entona��o.

Ap�s a escolha de um dataset adequado, pode-se desenvolver ou aprimorar
um modelo de RNA para uma classifica��o mais conveniente.

O c�digo fonte utilizado para a realiza��o desta pesquisa est�
dispon�vel em \[Omitido - Revis�o �s cegas\]

Metodologia {#sec:metodo}
===========

A metodologia empregada neste trabalho consistiu em cinco atividades
b�sicas:

1.  Sele��o e an�lise de dados: Nesta parte foram realizadas pesquisas
    de poss�veis �udios adequados ao problema e em seguida foram
    realizadas an�lises desses �udios para verificar quais eram as
    caracter�sticas originais dos dados obtidos, isto �, a dura��o total
    das inst�ncias, a taxa de amostragem original, a quantidade de
    inst�ncia de cada idioma, a presen�a ou n�o de ru�dos, a quantidade
    de locutores e a quantidade de locutores dos g�neros masculino e
    feminino.

2.  Prepara��o e valida��o das bases experimentais: Nesta etapa foram
    preparadas as bases utilizadas nos experimentos atrav�s do uso de um
    script para o pr�-processamento dos �udios. Foram geradas algumas
    bases com caracter�sticas chave tais como utiliza��o ou n�o de
    aumento de dados e remo��o ou n�o de sil�ncio, diferentes algoritmos
    de normaliza��o, entre outras. Cada base gerada foi validada atrav�s
    de uma an�lise detalhada verificando a taxa de amostragem, a
    quantidade de �udios total e de cada classe, a quantidade de
    locutores, al�m da plotagem de espectrogramas e da verifica��o dos
    dados aumentados.

3.  Prepara��o do modelo: Um modelo de rede neural convolucional foi
    preparado para a realiza��o dos experimentos. O aperfei�oamento e a
    busca por hiper-par�metros do modelo n�o foi objetivo do trabalho,
    mas procurou-se o desenvolvimento de um modelo capaz de atender aos
    objetivos propostos. O modelo convolucional foi escolhido por ser um
    tipo de rede neural robusto o suficiente para identificar as
    caracter�sticas dos fonemas intr�nsecos de cada idioma, por ser um
    tipo de rede cujo treinamento apresenta um desempenho adequado no
    hardware utilizado, por ser um tipo de rede j� estado-da-arte em
    algumas tarefas como reconhecimento de idioma [@bartz2017language],
    e tamb�m porque outras topologias convolucionais e MPL testadas n�o
    convergiram.

4.  Elabora��o e execu��o dos experimentos: Foram elaborados dois
    experimentos iniciais, seguido de um terceiro experimento para a
    realiza��o de conclus�es mais s�lidas. Cada experimento consistiu na
    execu��o do algoritmo de treinamento do modelo de rede neural
    convolucional proposto seguido de duas avalia��es de resultados
    utilizando as respectivas bases experimentais de cada dataset na
    �ltima �poca e na melhor �poca. As caracter�sticas dos �udios foram
    extra�das nesta etapa, logo antes dos dados serem consumidos pelo
    algoritmo de treinamento. Tamb�m foram armazenados os valores das
    acur�cias de valida��o de cada �poca de cada treinamento para
    posterior an�lise.

5.  Testes finais e an�lise dos resultados: Por �ltimo foram realizados
    testes finais em datasets chave, utilizando diferentes bases
    experimentais para a verifica��o do desempenho de cada modelo. Ap�s
    apontar os resultados, pode-se analisar com mais profundidade o
    impacto e a qualidade dos datasets utilizados nos experimentos.

Sele��o e an�lise dos dados
---------------------------

A primeira etapa no processo de desenvolvimento do sistema consistiu na
sele��o e an�lise de arquivos de �udio para a gera��o das bases
experimentais, o pr�-processamento dos dados e a realiza��o dos
experimentos. Inicialmente procurou-se a obten��o de uma base pr�pria
para este fim, ou uma base de discursos que contivesse todos os idiomas
alvo [^2] para o treinamento dos modelos.

N�o foram encontradas bases de discursos contendo todos os idiomas,
entretanto foram encontradas diversas bases de dados contendo um �nico
idioma. Al�m disso, algumas bases s�o pagas, o que inviabilizaria a sua
utiliza��o, j� que a tarefa de reconhecer o idioma n�o requer,
necessariamente, a rotula��o das falas [^3] e existem bases gratuitas ou
ainda outras fontes de �udio que poderiam ser utilizadas. Detalhes dos
�udios obtidos est�o dispon�veis no reposit�rio do projeto.

Para praticidade na pesquisa, este conjunto de dados foi denominado CPD
(*Corpus Dataset*) [^4].

A pesquisa de trabalhos relacionados e a motiva��o em selecionar os
dados adequados motivou a obten��o de um segundo conjunto de dados. Em
um reposit�rio do GitHub [^5] , o autor [@oponowicz] utilizou �udio
livros para a tarefa de reconhecimento de idiomas. Partindo deste
trabalho, o segundo conjunto de dados foi formado por �udio livros
obtidos a partir de grava��es *LibriVox*[^6]. Detalhes dos �udios
obtidos est�o dispon�veis no reposit�rio do projeto.

Novamente para a praticidade na pesquisa, este conjunto de dados foi
denominado SLD (*Spoken Language Dataset*) [^7].

A base CPD n�o apresenta separa��o de dados de treinamento e teste
originalmente, por isso, o conjunto de teste foi gerado manualmente. A
base SLD foi dividida em conjuntos de treinamento de teste e treinamento
de forma que dois �udios de sexos diferentes fossem destinados a teste
enquanto os demais fossem destinados ao conjunto de treinamento.

A base CPD obtida durante a realiza��o do trabalho e sem
pr�-processamento apresentou um tamanho aproximado de 21Gb, ao passo que
a base SLD apresentou um tamanho aproximado de 1,5Gb. Logo, poderia ser
computacionalmente invi�vel a realiza��o de m�ltiplos experimentos com
todos os �udios obtidos. Por isso, a dura��o total das bases foi
limitada e duas amostras menores dos dois conjuntos de dados foram
criados, da seguinte forma:

-   Para a base CPD foram selecionados pelo menos 1000 �udios com
    dura��o m�nima de 5.1 segundos de forma rand�mica para o conjunto de
    treinamento e pelo menos 100 �udios com dura��o m�nima de 5.1
    segundos de forma rand�mica para o conjunto de teste. As
    caracter�sticas originais dos �udios foram mantidas. Os �udios foram
    cortados para terem dura��o m�xima de 5.1 segundos.

-   Para a base SLD todos os �udios dos conjuntos de teste e treinamento
    foram considerados. Os �udios foram cortados para terem uma dura��o
    m�xima de 2 minutos.

O motivo pelo qual a dura��o dos �udios em SLD em compara��o com o CPD
serem t�o diferentes � o fato de que em CPD existe uma grande quantidade
de �udios de pequena dura��o, enquanto em SLD existem poucos �udios, mas
de longa dura��o. A dura��o dos �udios em CPD utilizada foi de 5.1
segundos porque neste trabalho ser�o utilizados trechos de �udio de 5
segundos para o treinamento do modelo, a dura��o extra de 100
milissegundos fornece uma seguran�a [^8] maior no processo de corte
posterior.

Ap�s a sele��o dos arquivos de �udio, as bases foram analisadas.

Para a an�lise dos dados foi utilizado a ferramenta *Jupyter* [^9]
contendo c�digos de an�lise de caracter�sticas importantes de cada
�udio. O objetivo era entender com mais profundidade os �udios coletados
e verificar a validade dos dados selecionados. Para isso, foram
carregados os �udios das bases e realizada uma checagem de certas
caracter�sticas como dura��o dos �udios e taxa de amostragem, e plotagem
de espectrogramas. Essa an�lise � importante para verificar a qualidade
dos dados obtidos e checar se o algoritmo implementado est� processando
os dados corretamente.

O print da an�lise feita pode ser visualizado nas figura
[\[fig:printJupyterRaw\]](#fig:printJupyterRaw){reference-type="ref"
reference="fig:printJupyterRaw"}. Os detalhes da an�lise est�o
dispon�veis no reposit�rio do projeto.

\centering
![Print da an�lise utilizando a ferramenta *Jupyter* das bases
selecionadas](print_raw.png){width=".8\textwidth"}

[\[fig:printJupyterRaw\]]{#fig:printJupyterRaw
label="fig:printJupyterRaw"}

Prepara��o e valida��o das bases experimentais
----------------------------------------------

Ap�s a sele��o dos dados deve-se pr�-processar (se necess�rio) estes
dados. No caso da base de �udios � fundamental pr�-processar o �udio
para garantir que todas as inst�ncias estejam com a mesma taxa de
amostragem, a mesma dura��o, e tamb�m que apresentem caracter�sticas
similares quanto a presen�a de ru�do e sil�ncio.

Usualmente apenas um dataset � pr�-processado e gerado e este �
suficiente para o treinamento do modelo. Entretanto, como o objetivo do
trabalho � analisar o impacto de diferentes configura��es dos �udios,
foram propostas algumas vers�es do dataset para a realiza��o dos
experimentos. Estes datasets foram ent�o gerados por um software
implementado pelo autor que utiliza ferramentas como a linguagem
*Python*[^10], a biblioteca *Librosa*[^11] e o software *Sox*[^12] para
processamento de �udio. As configura��es propostas podem ser vistas nas
tabelas
[\[tab:languagesByDataset\]](#tab:languagesByDataset){reference-type="ref"
reference="tab:languagesByDataset"},
[\[tab:effectsByDataset\]](#tab:effectsByDataset){reference-type="ref"
reference="tab:effectsByDataset"} e
[\[tab:augmentedByDataset\]](#tab:augmentedByDataset){reference-type="ref"
reference="tab:augmentedByDataset"}. Para fins pr�ticos da pesquisa,
foram denominadas vers�es de cada dataset gerado.

\centering
\vspace{.6cm}
  --------- -------- ---- ---- ---- ---- ----
   Dataset   Vers�o                      
                      PT   ES   DE   EN   FR
     CPD      0.1     x    x         x   
     CPD      0.2     x    x         x   
     CPD      0.3     x    x         x   
     CPD      1.1     x    x         x   
     CPD      2.1     x    x         x   
     SLD      0.1     x    x    x    x    x
     SLD      0.2     x    x    x    x    x
     SLD      0.3     x    x    x    x    x
     SLD      0.4     x    x    x    x    x
     SLD      0.5     x    x    x    x    x
     SLD      0.6     x    x    x    x    x
     SLD      1.0     x    x    x    x    x
     SLD      1.1     x    x    x    x    x
     SLD      1.2     x    x    x    x    x
     SLD      1.3     x    x    x    x    x
     SLD      1.4     x    x    x    x    x
     SLD      1.5     x    x    x    x    x
     SLD      2.0     x    x         x   
  --------- -------- ---- ---- ---- ---- ----

  : Idiomas por dataset

[\[tab:languagesByDataset\]]{#tab:languagesByDataset
label="tab:languagesByDataset"}

\vspace{.6cm}
    Dataset    Vers�o    Remo��o de sil.\*    Norm.\*\*    Filtro p. baixa    Taxa Amos.
  ---------- --------- -------------------- ------------ ------------------ -------------
     CPD        0.1            N�o             Padr�o           N�o             8000
     CPD        0.2            N�o             Padr�o           N�o             16000
     CPD        0.3            N�o             Padr�o           N�o             48000
     CPD        1.1            N�o             Padr�o           N�o             8000
     CPD        2.1            N�o             Padr�o           N�o             8000
     SLD        0.1            N�o             Padr�o           N�o             8000
     SLD        0.2            N�o             Padr�o           N�o             16000
     SLD        0.3            N�o             Padr�o           N�o             16000
     SLD        0.4            N�o             ITU-R            N�o             16000
     SLD        0.5            N�o              Peak            N�o             16000
     SLD        0.6            N�o             Padr�o           N�o             48000
     SLD        1.0             1%             Padr�o           N�o             16000
     SLD        1.1            N�o             Padr�o           N�o             16000
     SLD        1.2            N�o             Padr�o           N�o             48000
     SLD        1.3            N�o             Padr�o           6000            16000
     SLD        1.4             1%             Padr�o           6000            16000
     SLD        1.5             1%             Padr�o           N�o             16000
     SLD        2.0            N�o             Padr�o           N�o             16000

  : Efeitos de pr�-processamento por dataset

[\[tab:effectsByDataset\]]{#tab:effectsByDataset
label="tab:effectsByDataset"}

\footnotesize{NOTAS:
        \\ {*}Remo��o de sil.: remo��o de sil�ncio caso o volume do trecho esteja a abaixo do n�vel informado.
        \\ {**}Normaliza��o: A implementa��o padr�o normaliza os �udios a partir da mesma m�dia de amplitude, \textit{peak} � uma implementa��o onde os �udios s�o normalizados com base na amplitude m�xima de cada �udio, ITU-R � a implementa��o de ITU-R BS.1770-4.
    }
\centering
\vspace{.6cm}
  ---------- -------- ------------ ------- ----------- -----------------
    Dataset   Vers�o                                   
                       Velocidade   Ru�do   Entona��o   Filtro P. Baixa
     CPD       0.1                                     
     CPD       0.2                                     
     CPD       0.3                                     
     CPD       1.1         x                    x      
     CPD       2.1         x          x         x      
     SLD       0.1                                     
     SLD       0.2                                     
     SLD       0.3         x                    x      
     SLD       0.4                                     
     SLD       0.5                                     
     SLD       0.6                                     
     SLD       1.0         x          x         x      
     SLD       1.1         x          x         x      
     SLD       1.2         x          x         x      
     SLD       1.3         x          x         x      
     SLD       1.4         x          x         x      
     SLD       1.5         x          x         x              x
     SLD       2.0                                     
  ---------- -------- ------------ ------- ----------- -----------------

  : Aumento de dados por dataset

[\[tab:augmentedByDataset\]]{#tab:augmentedByDataset
label="tab:augmentedByDataset"}

� claro que nem todas as combina��es poss�veis poderiam ser testadas.
Entretanto, espera-se que as configura��es propostas atendam aos
objetivos do trabalho de forma satisfat�ria, sem comprometer a
viabilidade do mesmo.

Uma nova an�lise foi feita sobre os datasets gerados, desta vez
considerando outros aspectos como quantidade de dados aumentados,
locutores e g�nero por idioma (se dispon�vel), dura��o total, etc. O
print de uma das analises realizadas pode ser visualizado na figura
[\[fig:printJupyter\]](#fig:printJupyter){reference-type="ref"
reference="fig:printJupyter"}. A an�lise completa pode ser vista no
reposit�rio do projeto.

� importante deixar claro que n�o foram aplicados t�cnicas de aumento de
dados nos conjuntos de teste, apenas nos conjuntos de treinamento. Os
conjuntos de teste entre as diferentes vers�es propostas s�o bastante
similares, mas � necess�rio aplicar os mesmos efeitos de
pr�-processamento para garantir que o modelo seja capaz de reconhecer o
dado apresentado. Por exemplo, a taxa de amostragem implica em tamanhos
diferentes de imagens de espectrogramas devido ao fato de que a
frequ�ncia de Nyquist determina que para n�o haver perdas em um sinal
discreto deve-se utilizar o dobro da frequ�ncia m�xima de um sinal
[@grenander1959probability]. Ao passo que o espectrograma � a
representa��o visual das frequ�ncias com rela��o ao tempo, �udios com
taxa de amostragem diferentes produzem, a priori, tamanho diferentes de
imagens.

\centering
![Print da an�lise utilizando a ferramenta *Jupyter* das bases
pr�-processadas](print.png){width=".8\textwidth"}

[\[fig:printJupyter\]]{#fig:printJupyter label="fig:printJupyter"}

Uma vez que a prepara��o e a valida��o das bases experimentais �
conclu�da, pode-se prosseguir para a prepara��o dos experimentos. A
se��o [4.3](#subsec:modelPrep){reference-type="ref"
reference="subsec:modelPrep"} discute a elabora��o do modelo de RNA para
a realiza��o dos experimentos.

Prepara��o do modelo {#subsec:modelPrep}
--------------------

Foram testados alguns modelos convolucionais, convolucionais-recorrentes
e MLP. Alguns modelos testados j� foram utilizados anteriormente em
trabalhos similares, como o modelo em [@oponowicz] e
[@bartz2017language]. Inicialmente os modelos preparados e treinados com
um dos datasets gerados utilizando diferentes atributos (espectrogramas
e coeficientes MFCCs) n�o convergiram. Ap�s algumas tentativas um modelo
baseado na topologia proposta por [@oponowicz] convergiu e ent�o foi
utilizado para a realiza��o dos experimentos. � interessante notar que
este trabalho n�o procurou otimizar o modelo proposto, portanto, uma
pesquisa por modelos mais promissores podem melhorar muito mais os
resultados obtidos.

O modelo proposto neste trabalho � uma rede convolucional com cinco
camadas convolucionais seguidas por ativa��es ReLU, *max pooling* de
tamanho 2x2 e stride igual a 2 e dropout de 0.2. O tamanho do kernel e o
n�mero de filtros de cada camada convolucional � igual a (3x3, 500),
(3x3, 500), (3x3, 500), (3x3, 64), (3x3, 128), respectivamente. As
camadas convolucionais s�o seguidas por uma camada densa com ativa��o
softmax.

O script principal para a execu��o dos experimentos implementa a
topologia da RNA utilizando a biblioteca *keras* [^13] e fun��es �teis
para carregamento dos dados e extra��o de caracter�sticas. As
caracter�sticas foram extra�das em tempo real durante a execu��o dos
treinamentos, o que significa que os �udios foram carregados e os
espectrogramas foram gerados diretamente para serem consumidos pelo
algoritmo de treinamento do modelo.

Elabora��o e execu��o dos experimentos
--------------------------------------

A elabora��o dos experimentos foi realizada considerando o tempo de
execu��o total de cada treinamento, a quantidade de dados por �poca e a
quantidade total de �pocas. Considerando estas restri��es, os seguintes
experimentos foram propostos:

-   Experimento 1: o experimento 1 realiza o treinamento do modelo para
    cada dataset configurando um tempo m�ximo de execu��o para cada
    treinamento, e considerando uma itera��o em que uma �poca �
    realizada completamente[^14]. Foram considerados tamb�m um total de
    100 �pocas [^15]. O tempo m�ximo configurado foi de 10800 segundos
    (tr�s horas). O prop�sito deste treinamento � avaliar de uma forma
    geral o comportamento do treinamento considerando a evolu��o do
    desempenho a cada �poca completa.

-   Experimento 2: o experimento 2 realiza o treinamento do modelo para
    cada dataset tamb�m configurando um tempo m�ximo de execu��o para
    cada treinamento, e considerando uma itera��o de tamanho igual a
    1000 batches. Foram considerados tamb�m um total de 100 �pocas. O
    tempo m�ximo configurado foi de 10800 segundos (tr�s horas). O
    prop�sito deste treinamento � avaliar de uma forma geral o
    comportamento do treinamento considerando a evolu��o do desempenho a
    cada 1000 batches de dados.

-   Experimento 3: o experimento 3 n�o estabelece nenhuma restri��o de
    tempo. Uma �poca � considerada por itera��o e um total de 8 �pocas
    foi considerada por treinamento. Neste cen�rio, o desempenho final �
    avaliado sem considerar o tempo necess�rio para se chegar a este
    resultado.

Todos os treinamentos foram executados de forma online, isto �, com
batch igual a 1, e uma taxa de aprendizagem igual a 0,0001.

Os experimentos foram realizados atrav�s de uma execu��o de um script em
bash contendo os experimentos em um servidor dispon�vel na \[Omitido --
revis�o �s cegas\] [^16]. Os arquivos de log e resultados completos
podem ser visualizados no reposit�rio do projeto.

Testes finais e an�lise dos resultados
--------------------------------------

Para a realiza��o dos testes finais cada modelo treinado foi avaliado
com a sua respectiva base de teste na sua melhor �poca e na sua �ltima
�poca. A �ltima �poca se refere ao modelo obtido ap�s a execu��o da
�ltima itera��o do treinamento antes deste terminar ou ser interrompido,
enquanto a melhor �poca corresponde ao modelo obtido durante qualquer
fase de treinamento cujo resultado de avalia��o no conjunto de valida��o
foi a maior atingida, t�cnica comumente chamada de *model checkpoint*.
Os modelos mais promissores foram avaliados uma segunda vez com um
segundo dataset de teste de popula��o diferente das bases utilizadas no
treinamento. Por exemplo, no caso do treinamento da base CPD, o segundo
teste pode ser realizado com uma base de teste similar do conjunto SLD,
permitindo uma avalia��o mais concreta do desempenho do modelo.

A an�lise dos resultados � feita observando o resultado dos testes
finais para cada base em cada experimento. Foram gerados gr�ficos das
acur�cias de valida��o para verificar quais bases apresentaram uma maior
curva no aprendizado e inferir a partir destes gr�ficos as
caracter�sticas mais promissoras aplicadas. Foram analisados tamb�m
quais bases tiveram uma satura��o mais r�pida e quais apresentaram uma
maior propens�o a s sobreajuste (*overfitting*), atrav�s da observa��o
dos resultados finais utilizando os conjuntos de teste. Foram analisados
tamb�m poss�veis bases viciosas e propensas ao aprendizado de
caracter�sticas irrelevantes.

A principal forma de analisar os resultados obtidos foi a plotagem dos
gr�ficos de acur�cias/perdas por �poca dos treinamentos realizados,
isolando as caracter�sticas de interesse e observando o comportamento do
gr�fico ao longo do tempo e a observa��o dos maiores valores obtidos nas
acur�cias durante as avalia��es finais nos modelos ao utilizar os
devidos datasets de teste. Os experimentos 1, 2 e 3 s�o complementares e
ajudam nas an�lises quando os dados de um dado experimento s�o
insuficientes.

Resultados e Discuss�es {#sec:resultados}
=======================

O experimento 1 mostrou de uma forma geral que algumas bases convergiram
muito mais rapidamente que outras. Algumas bases apresentaram uma certa
converg�ncia precoce j� na primeira �poca, o que pode indicar problemas
de sobreajuste, j� que � poss�vel que o n�mero de filtros da topologia
proposta seja superior ao adequado.

Na figura
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"} � poss�vel visualizar a evolu��o da
acur�cia de valida��o ao longo das �pocas (o gr�fico da direita est� em
escala logar�tmica para facilitar a visualiza��o), algumas bases
apresentaram uma capacidade muito grande de aprendizado e outras
demoraram um pouco mais para convergir. De uma forma geral, no entanto,
todas as bases convergiram de uma forma ou de outra. Se o resultado do
teste final for positivo para todas as bases experimentadas, ent�o �
poss�vel que a RNN tenha uma capacidade at� surpreendente de lidar com
diferentes caracter�sticas do �udio e abstrair conceitos de forma
similar a um c�rebro humano, do contr�rio, � poss�vel que algumas das
bases experimentadas sustentem v�cios ao aprendizado do modelo e
deveriam ser corrigidas de alguma forma. A figura
[\[fig:exp1\_epoch\_loss\]](#fig:exp1_epoch_loss){reference-type="ref"
reference="fig:exp1_epoch_loss"} complementa o gr�fico
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"} com os valores de perda ao longo das
�pocas.

\centering
![Acur�cia de Valida��o por �poca e bases de dados no experimento
1](exp1_epoch_acc.png){width=".8\textwidth"}

[\[fig:exp1\_epoch\_acc\]]{#fig:exp1_epoch_acc
label="fig:exp1_epoch_acc"}

\centering
![Perda (loss) de Valida��o por �poca e bases de dados no experimento
1](exp1_epoch_loss.png){width=".8\textwidth"}

[\[fig:exp1\_epoch\_loss\]]{#fig:exp1_epoch_loss
label="fig:exp1_epoch_loss"}

\centering
![Compara��o de desempenho considerando remo��o de sil�ncio,
normaliza��o e filtro passa-baixa no experimento
1](exp1_epoch_acc_preprocessing.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_preprocessing\]]{#fig:exp1_epoch_acc_preprocessing
label="fig:exp1_epoch_acc_preprocessing"}

Ao plotar o gr�fico das acur�cias das valida��es destacando os efeitos
de pr�-processamento utilizados, verifica-se que os efeitos alternativos
de normaliza��o testados retardaram um pouco o aprendizado. Outro fato
observado � que os efeitos de remo��o de sil�ncio e filtro passa baixa
n�o produziram nenhum impacto significativo nos conjuntos de valida��o.
O gr�fico pode ser visualizado na figura
[\[fig:exp1\_epoch\_acc\_preprocessing\]](#fig:exp1_epoch_acc_preprocessing){reference-type="ref"
reference="fig:exp1_epoch_acc_preprocessing"}. � prov�vel que estes
efeitos de pr�-processamento utilizados sejam at� mesmo desnecess�rios.
No gr�fico, a cor azul representa as bases com normaliza��o diferentes
da padr�o, a cor amarela representa a utiliza��o de filtro passa baixa e
o triangulo representa a utiliza��o de remo��o de sil�ncio.

Outro fato interessante observado se deve a taxa de amostragem
utilizada. De acordo com o desempenho mostrado na figura
[\[fig:exp1\_epoch\_acc\_rate\]](#fig:exp1_epoch_acc_rate){reference-type="ref"
reference="fig:exp1_epoch_acc_rate"}, a taxa de amostragem n�o
proporcionou nenhum ganho ou nenhuma perda significativa quanto a
acur�cia do modelo, apenas quanto ao tempo necess�rio de treinamento
(devido principalmente ao tamanho da imagem gerada). No gr�fico da
figura
[\[fig:exp1\_epoch\_acc\_rate\]](#fig:exp1_epoch_acc_rate){reference-type="ref"
reference="fig:exp1_epoch_acc_rate"}, a cor cinza representa uma taxa de
amostragem de 8khz, a cor vermelha uma taxa de amostragem de 16khz e a
cor azul, uma taxa de 48khz. Isso pode ter ocorrido devido as pr�prias
condi��es originais dos �udios [^17], j� que um �udio originalmente com
uma taxa de 16khz n�o apresentar� ganho de qualidade se for convertido
para uma taxa superior, entretanto, os modelos treinados com �udios de
taxas maiores demandam mais recursos computacionais e obter �udios de
extrema qualidade pode ser uma tarefa dif�cil.

\centering
![Compara��o de desempenho considerando a taxa de amostragem no
experimento 1](exp1_epoch_acc_rate.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_rate\]]{#fig:exp1_epoch_acc_rate
label="fig:exp1_epoch_acc_rate"}

Uma das tarefas realizadas que realmente melhorou o desempenho dos
treinamentos foi a t�cnica de aumento de dados, ao observar a figura
[\[fig:exp1\_epoch\_acc\_aug\]](#fig:exp1_epoch_acc_aug){reference-type="ref"
reference="fig:exp1_epoch_acc_aug"}, percebe-se que os dados criados
artificialmente pelo t�cnica parecem se comportar da mesma forma como os
dados originais, possibilitando um poder maior de treinamento aos
modelos. Na figura
[\[fig:exp1\_epoch\_acc\_aug\]](#fig:exp1_epoch_acc_aug){reference-type="ref"
reference="fig:exp1_epoch_acc_aug"}, a cor azul representa o uso de
dados artificiais durante o treinamento e a cor cinza a aus�ncia da
t�cnica, o triangulo representa as bases do conjunto CPD e o c�rculo as
bases do conjunto SLD. Os treinamentos com as bases SLD tiveram um
aumento muito consider�vel, enquanto em CPD a melhora de desempenho foi
mais sutil.

\centering
![Compara��o de desempenho considerando aumento de dados no experimento
1](exp1_epoch_acc_aug.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_aug\]]{#fig:exp1_epoch_acc_aug
label="fig:exp1_epoch_acc_aug"}

Das t�cnicas de aumento de dados empregadas, a utiliza��o de adi��o de
ru�do pareceu fornecer pouca melhora no desempenho do treinamento. Na
figura
[\[fig:exp1\_epoch\_acc\_ns\]](#fig:exp1_epoch_acc_ns){reference-type="ref"
reference="fig:exp1_epoch_acc_ns"}, todas as bases mostradas apresentam
aumento de dados, a cor cinza significa aus�ncia de adi��o de ru�do.

\centering
![Compara��o de desempenho considerando adi��o de ru�do no experimento
1](exp1_epoch_acc_ns.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_ns\]]{#fig:exp1_epoch_acc_ns
label="fig:exp1_epoch_acc_ns"}

No experimento 2 a t�cnica de aumento de dados � colocada em
desvantagem. Como o experimento foi realizado de forma que cada itera��o
� realizada com a mesma quantidade de dados para todos os treinamentos,
a maior propor��o dos dados das bases com aumento de dados � artificial.
Espera-se portanto um comportamento pior ou similar ao ilustrado na
figura
[\[fig:augmented\_vs\_real\]](#fig:augmented_vs_real){reference-type="ref"
reference="fig:augmented_vs_real"}. De fato, o processo de aumento de
dados funcionou como o esperado, conforme pode ser visto na figura
[\[fig:exp2\_epoch\_acc\_aug\]](#fig:exp2_epoch_acc_aug){reference-type="ref"
reference="fig:exp2_epoch_acc_aug"}, o desempenho das bases com dados
artificiais ficou muito pr�ximo do desempenho das bases contendo apenas
dados originais.

\centering
![Desempenho esperado comparando o uso de dados aumentados e dados
reais](augmented_vs_real.pdf){width=".5\textwidth"}

[\[fig:augmented\_vs\_real\]]{#fig:augmented_vs_real
label="fig:augmented_vs_real"}

\centering
![Compara��o de desempenho considerando aumento de dados no experimento
2](exp2_epoch_acc_aug.png){width="100%"}

[\[fig:exp2\_epoch\_acc\_aug\]]{#fig:exp2_epoch_acc_aug
label="fig:exp2_epoch_acc_aug"}

Os modelos treinados durante o experimento 1 foram avaliados na melhor e
�ltima �poca, os resultados podem ser visualizados na tabela
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"}. A princ�pio, as bases CPD tiveram um
resultado muito superior as bases SLD.

\vspace{.6cm}
   Base de treinamento\*   Base de teste\*\*   Melhor �poca   �ltima �poca
  ----------------------- ------------------- -------------- --------------
       TR-CPD\_v0-1          TE-CPD\_v0-1         0,920          0,947
       TR-CPD\_v0-2          TE-CPD\_v0-2         0,933          0,940
       TR-CPD\_v0-3          TE-CPD\_v0-3         0,940          0,940
       TR-CPD\_v1-1          TE-CPD\_v1-1         0,963          0,950
       TR-CPD\_v2-1          TE-CPD\_v2-1         0,957          0,957
       TR-SLD\_v0-1          TE-SLD\_v0-1         0,635          0,596
       TR-SLD\_v0-2          TE-SLD\_v0-2         0,561          0,535
       TR-SLD\_v0-3          TE-SLD\_v0-3         0,604          0,565
       TR-SLD\_v0-4          TE-SLD\_v0-4         0,587          0,635
       TR-SLD\_v0-5          TE-SLD\_v0-5         0,604          0,600
       TR-SLD\_v0-6          TE-SLD\_v0-6         0,404          0,417
       TR-SLD\_v1-0          TE-SLD\_v1-0         0,637          0,637
       TR-SLD\_v1-1          TE-SLD\_v1-1         0,561          0,561
       TR-SLD\_v1-2          TE-SLD\_v1-2         0,509          0,509
       TR-SLD\_v1-3          TE-SLD\_v1-3         0,587          0,587
       TR-SLD\_v1-4          TE-SLD\_v1-4         0,591          0,591
       TR-SLD\_v1-5          TE-SLD\_v1-5         0,595          0,595
       TR-SLD\_v2-0          TE-SLD\_v2-0         0,841          0,848

  : Avalia��o dos modelos no experimento 1 utilizando datasets de teste
  na melhor e �ltima �poca

[\[tab:eval\_exp\_1\]]{#tab:eval_exp_1 label="tab:eval_exp_1"}

\footnotesize{NOTAS:
            \\ {*}Base de treinamento: o conjunto de treinamento (TR) representa a maior parcela do conjunto de dados e tamb�m incluem os conjuntos de valida��o nestes casos.
            \\ {**}Base de teste: o conjunto de teste (TE) representa a menor parcela do conjunto de dados. Os dados de teste n�o foram utilizadas no processo de treinamento (incluindo a valida��o entre �pocas do modelo).
        }
Comparando com o resultado obtido no conjunto de valida��o, a base SLD
teve um desempenho pior. Isso sugere uma propens�o maior ao sobreajuste,
� possivel que a raz�o seja o fato de existirem poucos locutores neste
dataset.

O experimento 2 n�o produziu modelos com desempenho significativo,
apenas as bases CPD tiveram um resultado similar ao experimento 1,
enquanto as bases SLD tiveram um desempenho ligeiramente pior,
principalmente devido ao fato da quantidade de �pocas ter sido
insuficiente para a obten��o de modelos mais adequados.

O experimento 3 apresentou resultados similares ao experimento 1.
Conforme pode ser visto na tabela
[\[tab:eval\_exp\_3\]](#tab:eval_exp_3){reference-type="ref"
reference="tab:eval_exp_3"}, os resultados das bases CPD pareceram ser
mais promissores. Al�m disso, neste experimento, alguns treinamentos n�o
convergiram. Por exemplo, o treinamento com a base SLD vers�o 1.1 n�o
aprendeu durante o processo. Isso pode significar que a remo��o de
sil�ncio contribui para o treinamento neste caso, j� que a �nica
diferen�a para a vers�o 1.0 � a aplica��o desta t�cnica, mas � prov�vel
que a escolha de uma arquitetura mais adequada de RNN produza resultados
melhores.

Na figura
[\[fig:exp3\_epoch\_acc\_aug\]](#fig:exp3_epoch_acc_aug){reference-type="ref"
reference="fig:exp3_epoch_acc_aug"} podem ser observados os desempenhos
das bases aumentadas. Os resultados foram similares aos obtidos nos
experimentos 1 e 2, exceto pelo modelo treinado com a base SLD vers�o
1.0, que n�o convergiu.

\centering
![Compara��o de desempenho considerando aumento de dados no experimento
3](exp3_epoch_acc_aug.png){width="100%"}

[\[fig:exp3\_epoch\_acc\_aug\]]{#fig:exp3_epoch_acc_aug
label="fig:exp3_epoch_acc_aug"}

Os modelos treinados durante o experimento 3 foram avaliados na melhor e
�ltima �poca, os resultados podem ser visualizados na tabela
[\[tab:eval\_exp\_3\]](#tab:eval_exp_3){reference-type="ref"
reference="tab:eval_exp_3"}.

\vspace{.6cm}
   Base de treinamento   Base de teste   Melhor �poca   �ltima �poca
  --------------------- --------------- -------------- --------------
      TR-CPD\_v0-1       TE-CPD\_v0-1       0,876          0,877
      TR-CPD\_v0-2       TE-CPD\_v0-2       0,837          0,837
      TR-CPD\_v0-3       TE-CPD\_v0-3       0,943          0,947
      TR-CPD\_v1-1       TE-CPD\_v1-1       0,950          0,967
      TR-CPD\_v2-1       TE-CPD\_v2-1       0,957          0,957
      TR-SLD\_v0-1       TE-SLD\_v0-1       0,365          0,365
      TR-SLD\_v0-2       TE-SLD\_v0-2       0,361          0,361
      TR-SLD\_v0-3       TE-SLD\_v0-3       0,670          0,522
      TR-SLD\_v0-4       TE-SLD\_v0-4       0,200          0,200
      TR-SLD\_v0-5       TE-SLD\_v0-5       0,361          0,361
      TR-SLD\_v0-6       TE-SLD\_v0-6       0,443          0,413
      TR-SLD\_v1-0       TE-SLD\_v1-0       0,665          0,642
      TR-SLD\_v1-1       TE-SLD\_v1-1       0,200          0,200
      TR-SLD\_v1-3       TE-SLD\_v1-3       0,674          0,626
      TR-SLD\_v1-4       TE-SLD\_v1-4       0,684          0,637
      TR-SLD\_v1-5       TE-SLD\_v1-5       0,679          0,637
      TR-SLD\_v2-0       TE-SLD\_v2-0       0,572          0,572

  : Avalia��o dos modelos no experimento 3 utilizando datasets de teste
  na melhor e �ltima �poca

[\[tab:eval\_exp\_3\]]{#tab:eval_exp_3 label="tab:eval_exp_3"}

Alguns modelos chave foram selecionados para a realiza��o dos testes
finais, isto �, os modelos foram selecionados para a realiza��o de
testes com bases de teste de popula��es distintas. O resultado pode ser
visto na tabela
[\[tab:eval\_final\]](#tab:eval_final){reference-type="ref"
reference="tab:eval_final"}

\vspace{.6cm}
   Experimento\*   Base de treinamento   Base de teste\*\*   Resultado
  --------------- --------------------- ------------------- -----------
         1            TR-CPD\_v0-2         TE-SLD\_v2-0        0,36
         1            TR-CPD\_v2-1         TE-SLD\_v0-1        0,52
         1            TR-SLD\_v1-0         TE-CPD\_v0-2        0,54
         1            TR-SLD\_v1-1         TE-CPD\_v0-2        0,58
         1            TR-SLD\_v2-0         TE-CPD\_v0-2        0,46

  : Avalia��o final

[\[tab:eval\_final\]]{#tab:eval_final label="tab:eval_final"}

\footnotesize{NOTAS:
            \\ {*} O modelo obtido no experimento informado.
            \\ {**} Foram selecionadas as bases com caracter�sticas mais pr�ximas de pr�-processamento.
        }
Os resultados das avalia��es nas tabelas
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"} e
[\[tab:eval\_final\]](#tab:eval_final){reference-type="ref"
reference="tab:eval_final"} mostram que tanto a base CPD e SLD podem ser
utilizadas para o treinamento de um modelo de RNA para o reconhecimento
de idioma, mas algumas precau��es devem ser tomadas. A principal tarefa
de pr�-processamento � a padroniza��o da taxa de amostragem, e, de
acordo com os resultados, � prov�vel que taxas menores, como 8khz, sejam
mais promissoras, principalmente em bases com qualidade mais baixa, como
� o caso da CPD.

Apesar de n�o terem sido observados resultados promissores para as bases
com taxas de amostragem inferiores nos testes da tabela
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"}, os testes finais em
[\[tab:eval\_final\]](#tab:eval_final){reference-type="ref"
reference="tab:eval_final"} sugerem que as bases com taxas maiores s�o
mais suscet�veis a v�cios no aprendizado. Isso pode ocorrer
principalmente devido ao fato das bases originais em certos idiomas
terem qualidade menor. Por isso, � razo�vel supor que � mais f�cil para
a rede aprender que um dado �udio pertence a um certo idioma baseando-se
na sua qualidade do que nos fonemas que caracterizam o dado idioma. Al�m
disso, uma taxa de amostragem menor proporciona um treinamento mais
r�pido, conforme observado na figura
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"}, os treinamentos com bases de taxas
menores alcan�aram um n�mero maior de �pocas.

Conclus�o {#sec:conc}
=========

Este trabalho possibilitou um maior entendimento quanto aos dados sendo
utilizados por um algoritmo de aprendizado de m�quina \"caixa preta\",
em que pode ser extremamente complicado entender o funcionamento interno
do algoritmo, e inferir se o modelo obtido � capaz de classificar
corretamente novas inst�ncias de �udio na �rea de reconhecimento de
idioma falado. � f�cil ver que certas t�cnicas adotadas podem ser
custosas computacionalmente, e dependendo da base utilizada, o modelo
pode apresentar v�cios, sofrer de sobreajuste, entre outros problemas.
Este trabalho tamb�m mostrou como a t�cnica de aumento de dados pode ser
�til em modelos de RNAs, melhorando significativamente o desempenho do
modelo treinado.

Quanto a sele��o de dados, fica claro a import�ncia de selecionar dados
com caracter�sticas similares entre as classes para evitar quaisquer
tipos de v�cios no treinamento. Como observado no trabalho, alguns
problemas podem ser tratados com o devido pr�-processamento.

Quanto as t�cnicas de pr�-processamento, observa-se que, apesar de serem
essenciais, uma vez garantido uma condi��o m�nima para que o modelo
aprenda, n�o existem mudan�as significativas de desempenho ou melhora na
taxa de aprendizado dependendo das condi��es de pr�-processamento
adotadas. De fato, algumas t�cnicas podem at� mesmo prejudicar o
desempenho e tornar o processo de treinamento muito custoso
computacionalmente. Levando em considera��o estes fatos, pode ser
extremamente interessante adotar medidas que facilitem o processo de
treinamento como a utiliza��o de uma menor taxa de amostragem o que
produz imagens com resolu��o menores. � poss�vel que a remo��o de
sil�ncio contribua levemente para o aprendizado, al�m de diminuir a
quantidade de dados insignificativos na base.

Com rela��o ao uso de t�cnicas de aumento de dados, observa-se uma
grande melhora na qualidade do dataset para o treinamento de modelos. De
fato, a t�cnica de aumento de dados pode evitar de forma significativa o
sobreajuste de modelos ao possibilitar que novas inst�ncias sejam
criadas artificialmente. Como observado no trabalho, os �udios criados
artificialmente se comportam de maneira muito pr�xima aos dados
originais, o que contribui para o treinamento do modelo de uma forma
geral.

Apesar da t�cnica de aumento de dados ter sido extremamente �til, n�o
ficou claro qual � o impacto significativo de cada t�cnica utilizada,
exceto pela adi��o de ru�do, que se mostrou pouco �til no processo.
Outro fator importante que ficou pouco claro � a qualidade das fontes
dos dados, neste trabalho foram testadas duas fontes diferentes, e ambos
conjuntos de bases testadas tiveram um desempenho aceit�vel.

Trabalhos futuros
-----------------

Para trabalhos futuros sugere-se a realiza��o de experimentos com
diferentes popula��es de dados, por exemplo, cole��es de not�cias do
*Youtube* (abordagem utilizada por [@bartz2017language]), grava��es de
r�dio e *podcasts*, e ainda, a realiza��o de experimentos com a jun��o
destas bases. Como bem observado por [@revay2019multiclass], uma das
grandes dificuldades no desenvolvimento de sistemas LID � o treinamento
de modelos partindo de um �nico dataset, o modelo obtido pode se
confudir quando testado com �udios com caracter�sticas distintas das
utilizadas no momento do treinamento.

Outra sugest�o � a realiza��o de experimentos com rela��o a extra��o de
caracter�sticas al�m da obten��o de espectrogramas para o treinamento de
modelos e tamb�m a realiza��o de experimentos com outros tipos de RNAs e
transfer�ncia de aprendizado.

[^1]: http://www.voxforge.org/

[^2]: O objetivo inicial do autor da pesquisa era o desenvolvimento de
    um sistema de reconhecimento autom�tico de idiomas nos idiomas
    ingl�s, espanhol e portugu�s. Mais tarde, foram adicionados os
    idiomas alem�o e franc�s em algumas das bases utilizadas.

[^3]: Algumas bases fornecem arquivos com a rotula��o da fala dos
    discursos de cada �udio, isto �, o texto sendo falado naquele
    momento. Estes arquivos s�o �teis para o treinamento de modelos de
    transcri��o autom�tica de discurso.

[^4]: O nome CPD foi arbitr�rio, mas a motiva��o do nome � que *Corpus*
    � o nome dado as bases de discursos pr�prias para pesquisas na �rea
    de reconhecimento de fala.

[^5]: O GitHub � uma plataforma de hospedagem de c�digo-fonte com
    controle de vers�o usando o Git e � uma ferramenta muito utilizada
    para projetos de software de qualquer tipo, incluindo c�digos-fonte
    de classificadores e sistemas inteligentes.

[^6]: https://librivox.org/

[^7]: Atualmente o SLD pr�-processado est� dispon�vel em
    https://www.kaggle.com/toponowicz/spoken-language-identification nos
    idiomas ingl�s, alem�o e espanhol. Os �udios desta base apresentam
    caracter�sticas similares as bases utilizadas neste trabalho,
    entretanto, as bases utilizadas neste trabalho foram geradas por um
    algoritmo pr�prio.

[^8]: Mais tarde foi observado que uma dura��o maior seria ainda mais
    adequada devido a aplica��o das t�cnicas de aumento de dados, j� que
    elas alteram a dura��o dos �udios.

[^9]: https://jupyter.org/

[^10]: https://www.python.org/

[^11]: https://librosa.github.io/librosa/index.html

[^12]: http://sox.sourceforge.net/

[^13]: https://keras.io/

[^14]: Usualmente uma �poca � completada por itera��o, mas pode-se
    modificar a quantidade de batches por itera��o de forma que sejam
    necess�rios mais ou menos itera��es para completar uma �poca.

[^15]: O n�mero de �pocas sugerido � bastante elevado porque neste
    experimento a ideia � interromper o treinamento ap�s ter decorrido
    uma certa quantidade de tempo.

[^16]: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 16Gb de RAM e NVIDIA
    TITAN V.

[^17]: Em SLD a maioria dos audios possu�am uma taxa de amostragem de
    22050Hz enquanto em CPD a maioria dos �udios possu�am uma qualidade
    de 16000hz. Apesar disto, as bases cujos �udios foram convertidos
    para uma taxa inferior apresentaram desempenho similar as bases com
    taxas de amostragem mais altas.
