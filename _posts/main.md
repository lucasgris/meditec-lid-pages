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
    Exploração e análise de datasets e técnicas de aumento de dados para
    identificação de idioma (LID) utilizando redes neurais artificiais
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

Redes Neurais Artificiais (RNAs) são algoritmos de aprendizado de
máquina \"caixa preta\", o que significa que o processo interno do
modelo ou sistema desenvolvido não é facilmente entendido, isso implica
que os dados representam um papel muito importante no treinamento de uma
RNA. Em especial, a escolha dos dados de entrada para o treinamento dos
modelos e o pré-processamento dos dados desempenham fator decisivo e
influenciam diretamente o desempenho do algoritmo. Neste trabalho, são
realizadas explorações de diferentes bases de áudios de discursos para a
identificação de idioma (LID) utilizando um modelo de RNA, analisando o
impacto de diferentes técnicas de pré-processamento e aumento de dados
no desempenho do modelo.

Introdução {#sec:intro}
==========

De uma forma geral, Redes Neurais Artificiais são máquinas projetadas
para *modelarem* a forma como o cérebro humano realiza uma tarefa
particular ou alguma função de interesse [@haykin2007redes]. De fato, o
uso de modelos de RNAs têm se tornado cada vez mais frequente em
diferentes campos de pesquisa, e se mostraram extremamente eficazes nas
mais diversas tarefas como reconhecimento de padrões, classificação de
imagens, visão computacional, transcrição de áudio, entre outras. Uma
etapa essencial no processo de desenvolvimento de sistemas inteligentes
é a seleção e o pré-processamento de dados a serem utilizados no
processo de aprendizado de máquina, e com as RNAs, esse processo pode
ser ainda mais importante. Neste trabalho, são explorados e analisados
diferentes técnicas quanto ao desenvolvimento de um dataset adequado ao
treinamento de um modelo de RNA para a identificação de idioma.

Uma das áreas de interesse das RNAs são aplicações em áudio. As
pesquisas na área tiveram início em 1952 a partir de um sistema
dependente de locutor que era capaz de reconhecer dígitos falados
[@juang2005automatic]. Pesquisas recentes na área de reconhecimento de
idioma incluem [@revay2019multiclass], [@bartz2017language],
[@richardson2015deep], [@zazo2016language] e [@montavon2009deep]. Nestes
trabalhos, diferentes abordagens foram utilizadas quanto a seleção de
bases de áudios, pré-processamento e extração de características.

No caso da identificação de idioma existem diversas bases de áudios que
podem ser utilizadas, como por exemplo, leituras de áudio livros ou
bases de discursos preparadas para tarefas relacionadas a reconhecimento
de discursos. Cada base apresenta seus fatores positivos e negativos.
Por exemplo, a leitura de um áudio livro é, em geral, feita apenas por
um locutor, o que implica em uma quantidade pobre de diferentes timbres
de vozes nos dados. Em contraste, uma base de discursos como a
LibriSpeech [@panayotov2015librispeech] ou a VoxForge [^1] podem
apresentar muito ruído ou certas características indesejadas. Além
disso, para o sucesso do aprendizado do modelo, é importante garantir
uma consistência nos dados entre as classes (no caso, os idiomas), caso
contrário o modelo poderá aprender conceitos desinteressantes para o
propósito do problema, o que gerará um modelo com pouca capacidade de
classificar corretamente novas instâncias.

A tarefa de pré-processamento dos áudios é fundamental, principalmente
porque certas características do áudio dependem da origem destes e do
pré-processamento já previamente realizado nestes dados, além de
conversões entre formatos, qualidade de gravação e presença ou não de
ruídos no momento da captação das vozes. No caso da utilização de bases
de discursos o pré-processamento pode desempenhar tarefa ainda mais
importante, principalmente porque cada base de discurso apresenta
características intrínsecas que devem ser verificadas e, se necessário,
corrigidas de alguma forma. Por exemplo, uma das bases utilizadas no
desenvolvimento dos dados de treinamento provém de gravações de rádio no
idioma espanhol [@hernandez2014ciempiess].

Outra tarefa importante no desenvolvimento de dados para a utilização em
modelos de aprendizado de máquina é a realização de técnicas de aumento
de dados. O aumento de dados consiste na geração de novas instâncias a
partir de dados antigos utilizando técnicas específicas. Em geral, o
aumento de dados promove um melhor desempenho no aprendizado do modelo
porque uma quantidade maior de dados promove uma melhor capacidade de
generalização e de aprendizado de características importantes. No mundo
do reconhecimento de imagens, uma das técnicas para aumento de dados é o
espelhamento, a rotação e a distorção. No caso dos áudios, podem ser
utilizadas técnicas como adição de ruído de fundo e mudança de entonação
e velocidade.

O objetivo geral deste artigo é analisar extensivamente alguns dos
possíveis conjuntos de dados a serem utilizados por um modelo de RNA a
fim de identificar com êxito a língua falada, partindo da seleção dos
dados até técnicas de pré-processamento e aumento de dados.

Espera-se avaliar o impacto dos dados no desenvolvimento de um modelo
RNA para identificação de idioma:

1.  Quanto a seleção dos dados, isto é, áudio livros ou bases de
    discursos nos idiomas alvo.

2.  Quanto as técnicas de pré-processamento de áudio, ou seja, a taxa de
    amostragem utilizada, a aplicação de filtro passa-baixa, a remoção
    de períodos longos de silêncio e a normalização de áudio.

3.  Quanto ao uso de técnicas de aumento de dados, no caso, a mudança de
    velocidade, a adição de ruído e/ou a mudança de entonação.

Após a escolha de um dataset adequado, pode-se desenvolver ou aprimorar
um modelo de RNA para uma classificação mais conveniente.

O código fonte utilizado para a realização desta pesquisa está
disponível em \[Omitido - Revisão às cegas\]

Metodologia {#sec:metodo}
===========

A metodologia empregada neste trabalho consistiu em cinco atividades
básicas:

1.  Seleção e análise de dados: Nesta parte foram realizadas pesquisas
    de possíveis áudios adequados ao problema e em seguida foram
    realizadas análises desses áudios para verificar quais eram as
    características originais dos dados obtidos, isto é, a duração total
    das instâncias, a taxa de amostragem original, a quantidade de
    instância de cada idioma, a presença ou não de ruídos, a quantidade
    de locutores e a quantidade de locutores dos gêneros masculino e
    feminino.

2.  Preparação e validação das bases experimentais: Nesta etapa foram
    preparadas as bases utilizadas nos experimentos através do uso de um
    script para o pré-processamento dos áudios. Foram geradas algumas
    bases com características chave tais como utilização ou não de
    aumento de dados e remoção ou não de silêncio, diferentes algoritmos
    de normalização, entre outras. Cada base gerada foi validada através
    de uma análise detalhada verificando a taxa de amostragem, a
    quantidade de áudios total e de cada classe, a quantidade de
    locutores, além da plotagem de espectrogramas e da verificação dos
    dados aumentados.

3.  Preparação do modelo: Um modelo de rede neural convolucional foi
    preparado para a realização dos experimentos. O aperfeiçoamento e a
    busca por hiper-parâmetros do modelo não foi objetivo do trabalho,
    mas procurou-se o desenvolvimento de um modelo capaz de atender aos
    objetivos propostos. O modelo convolucional foi escolhido por ser um
    tipo de rede neural robusto o suficiente para identificar as
    características dos fonemas intrínsecos de cada idioma, por ser um
    tipo de rede cujo treinamento apresenta um desempenho adequado no
    hardware utilizado, por ser um tipo de rede já estado-da-arte em
    algumas tarefas como reconhecimento de idioma [@bartz2017language],
    e também porque outras topologias convolucionais e MPL testadas não
    convergiram.

4.  Elaboração e execução dos experimentos: Foram elaborados dois
    experimentos iniciais, seguido de um terceiro experimento para a
    realização de conclusões mais sólidas. Cada experimento consistiu na
    execução do algoritmo de treinamento do modelo de rede neural
    convolucional proposto seguido de duas avaliações de resultados
    utilizando as respectivas bases experimentais de cada dataset na
    última época e na melhor época. As características dos áudios foram
    extraídas nesta etapa, logo antes dos dados serem consumidos pelo
    algoritmo de treinamento. Também foram armazenados os valores das
    acurácias de validação de cada época de cada treinamento para
    posterior análise.

5.  Testes finais e análise dos resultados: Por último foram realizados
    testes finais em datasets chave, utilizando diferentes bases
    experimentais para a verificação do desempenho de cada modelo. Após
    apontar os resultados, pode-se analisar com mais profundidade o
    impacto e a qualidade dos datasets utilizados nos experimentos.

Seleção e análise dos dados
---------------------------

A primeira etapa no processo de desenvolvimento do sistema consistiu na
seleção e análise de arquivos de áudio para a geração das bases
experimentais, o pré-processamento dos dados e a realização dos
experimentos. Inicialmente procurou-se a obtenção de uma base própria
para este fim, ou uma base de discursos que contivesse todos os idiomas
alvo [^2] para o treinamento dos modelos.

Não foram encontradas bases de discursos contendo todos os idiomas,
entretanto foram encontradas diversas bases de dados contendo um único
idioma. Além disso, algumas bases são pagas, o que inviabilizaria a sua
utilização, já que a tarefa de reconhecer o idioma não requer,
necessariamente, a rotulação das falas [^3] e existem bases gratuitas ou
ainda outras fontes de áudio que poderiam ser utilizadas. Detalhes dos
áudios obtidos estão disponíveis no repositório do projeto.

Para praticidade na pesquisa, este conjunto de dados foi denominado CPD
(*Corpus Dataset*) [^4].

A pesquisa de trabalhos relacionados e a motivação em selecionar os
dados adequados motivou a obtenção de um segundo conjunto de dados. Em
um repositório do GitHub [^5] , o autor [@oponowicz] utilizou áudio
livros para a tarefa de reconhecimento de idiomas. Partindo deste
trabalho, o segundo conjunto de dados foi formado por áudio livros
obtidos a partir de gravações *LibriVox*[^6]. Detalhes dos áudios
obtidos estão disponíveis no repositório do projeto.

Novamente para a praticidade na pesquisa, este conjunto de dados foi
denominado SLD (*Spoken Language Dataset*) [^7].

A base CPD não apresenta separação de dados de treinamento e teste
originalmente, por isso, o conjunto de teste foi gerado manualmente. A
base SLD foi dividida em conjuntos de treinamento de teste e treinamento
de forma que dois áudios de sexos diferentes fossem destinados a teste
enquanto os demais fossem destinados ao conjunto de treinamento.

A base CPD obtida durante a realização do trabalho e sem
pré-processamento apresentou um tamanho aproximado de 21Gb, ao passo que
a base SLD apresentou um tamanho aproximado de 1,5Gb. Logo, poderia ser
computacionalmente inviável a realização de múltiplos experimentos com
todos os áudios obtidos. Por isso, a duração total das bases foi
limitada e duas amostras menores dos dois conjuntos de dados foram
criados, da seguinte forma:

-   Para a base CPD foram selecionados pelo menos 1000 áudios com
    duração mínima de 5.1 segundos de forma randômica para o conjunto de
    treinamento e pelo menos 100 áudios com duração mínima de 5.1
    segundos de forma randômica para o conjunto de teste. As
    características originais dos áudios foram mantidas. Os áudios foram
    cortados para terem duração máxima de 5.1 segundos.

-   Para a base SLD todos os áudios dos conjuntos de teste e treinamento
    foram considerados. Os áudios foram cortados para terem uma duração
    máxima de 2 minutos.

O motivo pelo qual a duração dos áudios em SLD em comparação com o CPD
serem tão diferentes é o fato de que em CPD existe uma grande quantidade
de áudios de pequena duração, enquanto em SLD existem poucos áudios, mas
de longa duração. A duração dos áudios em CPD utilizada foi de 5.1
segundos porque neste trabalho serão utilizados trechos de áudio de 5
segundos para o treinamento do modelo, a duração extra de 100
milissegundos fornece uma segurança [^8] maior no processo de corte
posterior.

Após a seleção dos arquivos de áudio, as bases foram analisadas.

Para a análise dos dados foi utilizado a ferramenta *Jupyter* [^9]
contendo códigos de análise de características importantes de cada
áudio. O objetivo era entender com mais profundidade os áudios coletados
e verificar a validade dos dados selecionados. Para isso, foram
carregados os áudios das bases e realizada uma checagem de certas
características como duração dos áudios e taxa de amostragem, e plotagem
de espectrogramas. Essa análise é importante para verificar a qualidade
dos dados obtidos e checar se o algoritmo implementado está processando
os dados corretamente.

O print da análise feita pode ser visualizado nas figura
[\[fig:printJupyterRaw\]](#fig:printJupyterRaw){reference-type="ref"
reference="fig:printJupyterRaw"}. Os detalhes da análise estão
disponíveis no repositório do projeto.

\centering
![Print da análise utilizando a ferramenta *Jupyter* das bases
selecionadas](print_raw.png){width=".8\textwidth"}

[\[fig:printJupyterRaw\]]{#fig:printJupyterRaw
label="fig:printJupyterRaw"}

Preparação e validação das bases experimentais
----------------------------------------------

Após a seleção dos dados deve-se pré-processar (se necessário) estes
dados. No caso da base de áudios é fundamental pré-processar o áudio
para garantir que todas as instâncias estejam com a mesma taxa de
amostragem, a mesma duração, e também que apresentem características
similares quanto a presença de ruído e silêncio.

Usualmente apenas um dataset é pré-processado e gerado e este é
suficiente para o treinamento do modelo. Entretanto, como o objetivo do
trabalho é analisar o impacto de diferentes configurações dos áudios,
foram propostas algumas versões do dataset para a realização dos
experimentos. Estes datasets foram então gerados por um software
implementado pelo autor que utiliza ferramentas como a linguagem
*Python*[^10], a biblioteca *Librosa*[^11] e o software *Sox*[^12] para
processamento de áudio. As configurações propostas podem ser vistas nas
tabelas
[\[tab:languagesByDataset\]](#tab:languagesByDataset){reference-type="ref"
reference="tab:languagesByDataset"},
[\[tab:effectsByDataset\]](#tab:effectsByDataset){reference-type="ref"
reference="tab:effectsByDataset"} e
[\[tab:augmentedByDataset\]](#tab:augmentedByDataset){reference-type="ref"
reference="tab:augmentedByDataset"}. Para fins práticos da pesquisa,
foram denominadas versões de cada dataset gerado.

\centering
\vspace{.6cm}
  --------- -------- ---- ---- ---- ---- ----
   Dataset   Versão                      
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
    Dataset    Versão    Remoção de sil.\*    Norm.\*\*    Filtro p. baixa    Taxa Amos.
  ---------- --------- -------------------- ------------ ------------------ -------------
     CPD        0.1            Não             Padrão           Não             8000
     CPD        0.2            Não             Padrão           Não             16000
     CPD        0.3            Não             Padrão           Não             48000
     CPD        1.1            Não             Padrão           Não             8000
     CPD        2.1            Não             Padrão           Não             8000
     SLD        0.1            Não             Padrão           Não             8000
     SLD        0.2            Não             Padrão           Não             16000
     SLD        0.3            Não             Padrão           Não             16000
     SLD        0.4            Não             ITU-R            Não             16000
     SLD        0.5            Não              Peak            Não             16000
     SLD        0.6            Não             Padrão           Não             48000
     SLD        1.0             1%             Padrão           Não             16000
     SLD        1.1            Não             Padrão           Não             16000
     SLD        1.2            Não             Padrão           Não             48000
     SLD        1.3            Não             Padrão           6000            16000
     SLD        1.4             1%             Padrão           6000            16000
     SLD        1.5             1%             Padrão           Não             16000
     SLD        2.0            Não             Padrão           Não             16000

  : Efeitos de pré-processamento por dataset

[\[tab:effectsByDataset\]]{#tab:effectsByDataset
label="tab:effectsByDataset"}

\footnotesize{NOTAS:
        \\ {*}Remoção de sil.: remoção de silêncio caso o volume do trecho esteja a abaixo do nível informado.
        \\ {**}Normalização: A implementação padrão normaliza os áudios a partir da mesma média de amplitude, \textit{peak} é uma implementação onde os áudios são normalizados com base na amplitude máxima de cada áudio, ITU-R é a implementação de ITU-R BS.1770-4.
    }
\centering
\vspace{.6cm}
  ---------- -------- ------------ ------- ----------- -----------------
    Dataset   Versão                                   
                       Velocidade   Ruído   Entonação   Filtro P. Baixa
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

É claro que nem todas as combinações possíveis poderiam ser testadas.
Entretanto, espera-se que as configurações propostas atendam aos
objetivos do trabalho de forma satisfatória, sem comprometer a
viabilidade do mesmo.

Uma nova análise foi feita sobre os datasets gerados, desta vez
considerando outros aspectos como quantidade de dados aumentados,
locutores e gênero por idioma (se disponível), duração total, etc. O
print de uma das analises realizadas pode ser visualizado na figura
[\[fig:printJupyter\]](#fig:printJupyter){reference-type="ref"
reference="fig:printJupyter"}. A análise completa pode ser vista no
repositório do projeto.

É importante deixar claro que não foram aplicados técnicas de aumento de
dados nos conjuntos de teste, apenas nos conjuntos de treinamento. Os
conjuntos de teste entre as diferentes versões propostas são bastante
similares, mas é necessário aplicar os mesmos efeitos de
pré-processamento para garantir que o modelo seja capaz de reconhecer o
dado apresentado. Por exemplo, a taxa de amostragem implica em tamanhos
diferentes de imagens de espectrogramas devido ao fato de que a
frequência de Nyquist determina que para não haver perdas em um sinal
discreto deve-se utilizar o dobro da frequência máxima de um sinal
[@grenander1959probability]. Ao passo que o espectrograma é a
representação visual das frequências com relação ao tempo, áudios com
taxa de amostragem diferentes produzem, a priori, tamanho diferentes de
imagens.

\centering
![Print da análise utilizando a ferramenta *Jupyter* das bases
pré-processadas](print.png){width=".8\textwidth"}

[\[fig:printJupyter\]]{#fig:printJupyter label="fig:printJupyter"}

Uma vez que a preparação e a validação das bases experimentais é
concluída, pode-se prosseguir para a preparação dos experimentos. A
seção [4.3](#subsec:modelPrep){reference-type="ref"
reference="subsec:modelPrep"} discute a elaboração do modelo de RNA para
a realização dos experimentos.

Preparação do modelo {#subsec:modelPrep}
--------------------

Foram testados alguns modelos convolucionais, convolucionais-recorrentes
e MLP. Alguns modelos testados já foram utilizados anteriormente em
trabalhos similares, como o modelo em [@oponowicz] e
[@bartz2017language]. Inicialmente os modelos preparados e treinados com
um dos datasets gerados utilizando diferentes atributos (espectrogramas
e coeficientes MFCCs) não convergiram. Após algumas tentativas um modelo
baseado na topologia proposta por [@oponowicz] convergiu e então foi
utilizado para a realização dos experimentos. É interessante notar que
este trabalho não procurou otimizar o modelo proposto, portanto, uma
pesquisa por modelos mais promissores podem melhorar muito mais os
resultados obtidos.

O modelo proposto neste trabalho é uma rede convolucional com cinco
camadas convolucionais seguidas por ativações ReLU, *max pooling* de
tamanho 2x2 e stride igual a 2 e dropout de 0.2. O tamanho do kernel e o
número de filtros de cada camada convolucional é igual a (3x3, 500),
(3x3, 500), (3x3, 500), (3x3, 64), (3x3, 128), respectivamente. As
camadas convolucionais são seguidas por uma camada densa com ativação
softmax.

O script principal para a execução dos experimentos implementa a
topologia da RNA utilizando a biblioteca *keras* [^13] e funções úteis
para carregamento dos dados e extração de características. As
características foram extraídas em tempo real durante a execução dos
treinamentos, o que significa que os áudios foram carregados e os
espectrogramas foram gerados diretamente para serem consumidos pelo
algoritmo de treinamento do modelo.

Elaboração e execução dos experimentos
--------------------------------------

A elaboração dos experimentos foi realizada considerando o tempo de
execução total de cada treinamento, a quantidade de dados por época e a
quantidade total de épocas. Considerando estas restrições, os seguintes
experimentos foram propostos:

-   Experimento 1: o experimento 1 realiza o treinamento do modelo para
    cada dataset configurando um tempo máximo de execução para cada
    treinamento, e considerando uma iteração em que uma época é
    realizada completamente[^14]. Foram considerados também um total de
    100 épocas [^15]. O tempo máximo configurado foi de 10800 segundos
    (três horas). O propósito deste treinamento é avaliar de uma forma
    geral o comportamento do treinamento considerando a evolução do
    desempenho a cada época completa.

-   Experimento 2: o experimento 2 realiza o treinamento do modelo para
    cada dataset também configurando um tempo máximo de execução para
    cada treinamento, e considerando uma iteração de tamanho igual a
    1000 batches. Foram considerados também um total de 100 épocas. O
    tempo máximo configurado foi de 10800 segundos (três horas). O
    propósito deste treinamento é avaliar de uma forma geral o
    comportamento do treinamento considerando a evolução do desempenho a
    cada 1000 batches de dados.

-   Experimento 3: o experimento 3 não estabelece nenhuma restrição de
    tempo. Uma época é considerada por iteração e um total de 8 épocas
    foi considerada por treinamento. Neste cenário, o desempenho final é
    avaliado sem considerar o tempo necessário para se chegar a este
    resultado.

Todos os treinamentos foram executados de forma online, isto é, com
batch igual a 1, e uma taxa de aprendizagem igual a 0,0001.

Os experimentos foram realizados através de uma execução de um script em
bash contendo os experimentos em um servidor disponível na \[Omitido --
revisão às cegas\] [^16]. Os arquivos de log e resultados completos
podem ser visualizados no repositório do projeto.

Testes finais e análise dos resultados
--------------------------------------

Para a realização dos testes finais cada modelo treinado foi avaliado
com a sua respectiva base de teste na sua melhor época e na sua última
época. A última época se refere ao modelo obtido após a execução da
última iteração do treinamento antes deste terminar ou ser interrompido,
enquanto a melhor época corresponde ao modelo obtido durante qualquer
fase de treinamento cujo resultado de avaliação no conjunto de validação
foi a maior atingida, técnica comumente chamada de *model checkpoint*.
Os modelos mais promissores foram avaliados uma segunda vez com um
segundo dataset de teste de população diferente das bases utilizadas no
treinamento. Por exemplo, no caso do treinamento da base CPD, o segundo
teste pode ser realizado com uma base de teste similar do conjunto SLD,
permitindo uma avaliação mais concreta do desempenho do modelo.

A análise dos resultados é feita observando o resultado dos testes
finais para cada base em cada experimento. Foram gerados gráficos das
acurácias de validação para verificar quais bases apresentaram uma maior
curva no aprendizado e inferir a partir destes gráficos as
características mais promissoras aplicadas. Foram analisados também
quais bases tiveram uma saturação mais rápida e quais apresentaram uma
maior propensão a s sobreajuste (*overfitting*), através da observação
dos resultados finais utilizando os conjuntos de teste. Foram analisados
também possíveis bases viciosas e propensas ao aprendizado de
características irrelevantes.

A principal forma de analisar os resultados obtidos foi a plotagem dos
gráficos de acurácias/perdas por época dos treinamentos realizados,
isolando as características de interesse e observando o comportamento do
gráfico ao longo do tempo e a observação dos maiores valores obtidos nas
acurácias durante as avaliações finais nos modelos ao utilizar os
devidos datasets de teste. Os experimentos 1, 2 e 3 são complementares e
ajudam nas análises quando os dados de um dado experimento são
insuficientes.

Resultados e Discussões {#sec:resultados}
=======================

O experimento 1 mostrou de uma forma geral que algumas bases convergiram
muito mais rapidamente que outras. Algumas bases apresentaram uma certa
convergência precoce já na primeira época, o que pode indicar problemas
de sobreajuste, já que é possível que o número de filtros da topologia
proposta seja superior ao adequado.

Na figura
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"} é possível visualizar a evolução da
acurácia de validação ao longo das épocas (o gráfico da direita está em
escala logarítmica para facilitar a visualização), algumas bases
apresentaram uma capacidade muito grande de aprendizado e outras
demoraram um pouco mais para convergir. De uma forma geral, no entanto,
todas as bases convergiram de uma forma ou de outra. Se o resultado do
teste final for positivo para todas as bases experimentadas, então é
possível que a RNN tenha uma capacidade até surpreendente de lidar com
diferentes características do áudio e abstrair conceitos de forma
similar a um cérebro humano, do contrário, é possível que algumas das
bases experimentadas sustentem vícios ao aprendizado do modelo e
deveriam ser corrigidas de alguma forma. A figura
[\[fig:exp1\_epoch\_loss\]](#fig:exp1_epoch_loss){reference-type="ref"
reference="fig:exp1_epoch_loss"} complementa o gráfico
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"} com os valores de perda ao longo das
épocas.

\centering
![Acurácia de Validação por época e bases de dados no experimento
1](exp1_epoch_acc.png){width=".8\textwidth"}

[\[fig:exp1\_epoch\_acc\]]{#fig:exp1_epoch_acc
label="fig:exp1_epoch_acc"}

\centering
![Perda (loss) de Validação por época e bases de dados no experimento
1](exp1_epoch_loss.png){width=".8\textwidth"}

[\[fig:exp1\_epoch\_loss\]]{#fig:exp1_epoch_loss
label="fig:exp1_epoch_loss"}

\centering
![Comparação de desempenho considerando remoção de silêncio,
normalização e filtro passa-baixa no experimento
1](exp1_epoch_acc_preprocessing.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_preprocessing\]]{#fig:exp1_epoch_acc_preprocessing
label="fig:exp1_epoch_acc_preprocessing"}

Ao plotar o gráfico das acurácias das validações destacando os efeitos
de pré-processamento utilizados, verifica-se que os efeitos alternativos
de normalização testados retardaram um pouco o aprendizado. Outro fato
observado é que os efeitos de remoção de silêncio e filtro passa baixa
não produziram nenhum impacto significativo nos conjuntos de validação.
O gráfico pode ser visualizado na figura
[\[fig:exp1\_epoch\_acc\_preprocessing\]](#fig:exp1_epoch_acc_preprocessing){reference-type="ref"
reference="fig:exp1_epoch_acc_preprocessing"}. É provável que estes
efeitos de pré-processamento utilizados sejam até mesmo desnecessários.
No gráfico, a cor azul representa as bases com normalização diferentes
da padrão, a cor amarela representa a utilização de filtro passa baixa e
o triangulo representa a utilização de remoção de silêncio.

Outro fato interessante observado se deve a taxa de amostragem
utilizada. De acordo com o desempenho mostrado na figura
[\[fig:exp1\_epoch\_acc\_rate\]](#fig:exp1_epoch_acc_rate){reference-type="ref"
reference="fig:exp1_epoch_acc_rate"}, a taxa de amostragem não
proporcionou nenhum ganho ou nenhuma perda significativa quanto a
acurácia do modelo, apenas quanto ao tempo necessário de treinamento
(devido principalmente ao tamanho da imagem gerada). No gráfico da
figura
[\[fig:exp1\_epoch\_acc\_rate\]](#fig:exp1_epoch_acc_rate){reference-type="ref"
reference="fig:exp1_epoch_acc_rate"}, a cor cinza representa uma taxa de
amostragem de 8khz, a cor vermelha uma taxa de amostragem de 16khz e a
cor azul, uma taxa de 48khz. Isso pode ter ocorrido devido as próprias
condições originais dos áudios [^17], já que um áudio originalmente com
uma taxa de 16khz não apresentará ganho de qualidade se for convertido
para uma taxa superior, entretanto, os modelos treinados com áudios de
taxas maiores demandam mais recursos computacionais e obter áudios de
extrema qualidade pode ser uma tarefa difícil.

\centering
![Comparação de desempenho considerando a taxa de amostragem no
experimento 1](exp1_epoch_acc_rate.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_rate\]]{#fig:exp1_epoch_acc_rate
label="fig:exp1_epoch_acc_rate"}

Uma das tarefas realizadas que realmente melhorou o desempenho dos
treinamentos foi a técnica de aumento de dados, ao observar a figura
[\[fig:exp1\_epoch\_acc\_aug\]](#fig:exp1_epoch_acc_aug){reference-type="ref"
reference="fig:exp1_epoch_acc_aug"}, percebe-se que os dados criados
artificialmente pelo técnica parecem se comportar da mesma forma como os
dados originais, possibilitando um poder maior de treinamento aos
modelos. Na figura
[\[fig:exp1\_epoch\_acc\_aug\]](#fig:exp1_epoch_acc_aug){reference-type="ref"
reference="fig:exp1_epoch_acc_aug"}, a cor azul representa o uso de
dados artificiais durante o treinamento e a cor cinza a ausência da
técnica, o triangulo representa as bases do conjunto CPD e o círculo as
bases do conjunto SLD. Os treinamentos com as bases SLD tiveram um
aumento muito considerável, enquanto em CPD a melhora de desempenho foi
mais sutil.

\centering
![Comparação de desempenho considerando aumento de dados no experimento
1](exp1_epoch_acc_aug.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_aug\]]{#fig:exp1_epoch_acc_aug
label="fig:exp1_epoch_acc_aug"}

Das técnicas de aumento de dados empregadas, a utilização de adição de
ruído pareceu fornecer pouca melhora no desempenho do treinamento. Na
figura
[\[fig:exp1\_epoch\_acc\_ns\]](#fig:exp1_epoch_acc_ns){reference-type="ref"
reference="fig:exp1_epoch_acc_ns"}, todas as bases mostradas apresentam
aumento de dados, a cor cinza significa ausência de adição de ruído.

\centering
![Comparação de desempenho considerando adição de ruído no experimento
1](exp1_epoch_acc_ns.png){width="100%"}

[\[fig:exp1\_epoch\_acc\_ns\]]{#fig:exp1_epoch_acc_ns
label="fig:exp1_epoch_acc_ns"}

No experimento 2 a técnica de aumento de dados é colocada em
desvantagem. Como o experimento foi realizado de forma que cada iteração
é realizada com a mesma quantidade de dados para todos os treinamentos,
a maior proporção dos dados das bases com aumento de dados é artificial.
Espera-se portanto um comportamento pior ou similar ao ilustrado na
figura
[\[fig:augmented\_vs\_real\]](#fig:augmented_vs_real){reference-type="ref"
reference="fig:augmented_vs_real"}. De fato, o processo de aumento de
dados funcionou como o esperado, conforme pode ser visto na figura
[\[fig:exp2\_epoch\_acc\_aug\]](#fig:exp2_epoch_acc_aug){reference-type="ref"
reference="fig:exp2_epoch_acc_aug"}, o desempenho das bases com dados
artificiais ficou muito próximo do desempenho das bases contendo apenas
dados originais.

\centering
![Desempenho esperado comparando o uso de dados aumentados e dados
reais](augmented_vs_real.pdf){width=".5\textwidth"}

[\[fig:augmented\_vs\_real\]]{#fig:augmented_vs_real
label="fig:augmented_vs_real"}

\centering
![Comparação de desempenho considerando aumento de dados no experimento
2](exp2_epoch_acc_aug.png){width="100%"}

[\[fig:exp2\_epoch\_acc\_aug\]]{#fig:exp2_epoch_acc_aug
label="fig:exp2_epoch_acc_aug"}

Os modelos treinados durante o experimento 1 foram avaliados na melhor e
última época, os resultados podem ser visualizados na tabela
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"}. A princípio, as bases CPD tiveram um
resultado muito superior as bases SLD.

\vspace{.6cm}
   Base de treinamento\*   Base de teste\*\*   Melhor época   Última época
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

  : Avaliação dos modelos no experimento 1 utilizando datasets de teste
  na melhor e última época

[\[tab:eval\_exp\_1\]]{#tab:eval_exp_1 label="tab:eval_exp_1"}

\footnotesize{NOTAS:
            \\ {*}Base de treinamento: o conjunto de treinamento (TR) representa a maior parcela do conjunto de dados e também incluem os conjuntos de validação nestes casos.
            \\ {**}Base de teste: o conjunto de teste (TE) representa a menor parcela do conjunto de dados. Os dados de teste não foram utilizadas no processo de treinamento (incluindo a validação entre épocas do modelo).
        }
Comparando com o resultado obtido no conjunto de validação, a base SLD
teve um desempenho pior. Isso sugere uma propensão maior ao sobreajuste,
é possivel que a razão seja o fato de existirem poucos locutores neste
dataset.

O experimento 2 não produziu modelos com desempenho significativo,
apenas as bases CPD tiveram um resultado similar ao experimento 1,
enquanto as bases SLD tiveram um desempenho ligeiramente pior,
principalmente devido ao fato da quantidade de épocas ter sido
insuficiente para a obtenção de modelos mais adequados.

O experimento 3 apresentou resultados similares ao experimento 1.
Conforme pode ser visto na tabela
[\[tab:eval\_exp\_3\]](#tab:eval_exp_3){reference-type="ref"
reference="tab:eval_exp_3"}, os resultados das bases CPD pareceram ser
mais promissores. Além disso, neste experimento, alguns treinamentos não
convergiram. Por exemplo, o treinamento com a base SLD versão 1.1 não
aprendeu durante o processo. Isso pode significar que a remoção de
silêncio contribui para o treinamento neste caso, já que a única
diferença para a versão 1.0 é a aplicação desta técnica, mas é provável
que a escolha de uma arquitetura mais adequada de RNN produza resultados
melhores.

Na figura
[\[fig:exp3\_epoch\_acc\_aug\]](#fig:exp3_epoch_acc_aug){reference-type="ref"
reference="fig:exp3_epoch_acc_aug"} podem ser observados os desempenhos
das bases aumentadas. Os resultados foram similares aos obtidos nos
experimentos 1 e 2, exceto pelo modelo treinado com a base SLD versão
1.0, que não convergiu.

\centering
![Comparação de desempenho considerando aumento de dados no experimento
3](exp3_epoch_acc_aug.png){width="100%"}

[\[fig:exp3\_epoch\_acc\_aug\]]{#fig:exp3_epoch_acc_aug
label="fig:exp3_epoch_acc_aug"}

Os modelos treinados durante o experimento 3 foram avaliados na melhor e
última época, os resultados podem ser visualizados na tabela
[\[tab:eval\_exp\_3\]](#tab:eval_exp_3){reference-type="ref"
reference="tab:eval_exp_3"}.

\vspace{.6cm}
   Base de treinamento   Base de teste   Melhor época   Última época
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

  : Avaliação dos modelos no experimento 3 utilizando datasets de teste
  na melhor e última época

[\[tab:eval\_exp\_3\]]{#tab:eval_exp_3 label="tab:eval_exp_3"}

Alguns modelos chave foram selecionados para a realização dos testes
finais, isto é, os modelos foram selecionados para a realização de
testes com bases de teste de populações distintas. O resultado pode ser
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

  : Avaliação final

[\[tab:eval\_final\]]{#tab:eval_final label="tab:eval_final"}

\footnotesize{NOTAS:
            \\ {*} O modelo obtido no experimento informado.
            \\ {**} Foram selecionadas as bases com características mais próximas de pré-processamento.
        }
Os resultados das avaliações nas tabelas
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"} e
[\[tab:eval\_final\]](#tab:eval_final){reference-type="ref"
reference="tab:eval_final"} mostram que tanto a base CPD e SLD podem ser
utilizadas para o treinamento de um modelo de RNA para o reconhecimento
de idioma, mas algumas precauções devem ser tomadas. A principal tarefa
de pré-processamento é a padronização da taxa de amostragem, e, de
acordo com os resultados, é provável que taxas menores, como 8khz, sejam
mais promissoras, principalmente em bases com qualidade mais baixa, como
é o caso da CPD.

Apesar de não terem sido observados resultados promissores para as bases
com taxas de amostragem inferiores nos testes da tabela
[\[tab:eval\_exp\_1\]](#tab:eval_exp_1){reference-type="ref"
reference="tab:eval_exp_1"}, os testes finais em
[\[tab:eval\_final\]](#tab:eval_final){reference-type="ref"
reference="tab:eval_final"} sugerem que as bases com taxas maiores são
mais suscetíveis a vícios no aprendizado. Isso pode ocorrer
principalmente devido ao fato das bases originais em certos idiomas
terem qualidade menor. Por isso, é razoável supor que é mais fácil para
a rede aprender que um dado áudio pertence a um certo idioma baseando-se
na sua qualidade do que nos fonemas que caracterizam o dado idioma. Além
disso, uma taxa de amostragem menor proporciona um treinamento mais
rápido, conforme observado na figura
[\[fig:exp1\_epoch\_acc\]](#fig:exp1_epoch_acc){reference-type="ref"
reference="fig:exp1_epoch_acc"}, os treinamentos com bases de taxas
menores alcançaram um número maior de épocas.

Conclusão {#sec:conc}
=========

Este trabalho possibilitou um maior entendimento quanto aos dados sendo
utilizados por um algoritmo de aprendizado de máquina \"caixa preta\",
em que pode ser extremamente complicado entender o funcionamento interno
do algoritmo, e inferir se o modelo obtido é capaz de classificar
corretamente novas instâncias de áudio na área de reconhecimento de
idioma falado. É fácil ver que certas técnicas adotadas podem ser
custosas computacionalmente, e dependendo da base utilizada, o modelo
pode apresentar vícios, sofrer de sobreajuste, entre outros problemas.
Este trabalho também mostrou como a técnica de aumento de dados pode ser
útil em modelos de RNAs, melhorando significativamente o desempenho do
modelo treinado.

Quanto a seleção de dados, fica claro a importância de selecionar dados
com características similares entre as classes para evitar quaisquer
tipos de vícios no treinamento. Como observado no trabalho, alguns
problemas podem ser tratados com o devido pré-processamento.

Quanto as técnicas de pré-processamento, observa-se que, apesar de serem
essenciais, uma vez garantido uma condição mínima para que o modelo
aprenda, não existem mudanças significativas de desempenho ou melhora na
taxa de aprendizado dependendo das condições de pré-processamento
adotadas. De fato, algumas técnicas podem até mesmo prejudicar o
desempenho e tornar o processo de treinamento muito custoso
computacionalmente. Levando em consideração estes fatos, pode ser
extremamente interessante adotar medidas que facilitem o processo de
treinamento como a utilização de uma menor taxa de amostragem o que
produz imagens com resolução menores. É possível que a remoção de
silêncio contribua levemente para o aprendizado, além de diminuir a
quantidade de dados insignificativos na base.

Com relação ao uso de técnicas de aumento de dados, observa-se uma
grande melhora na qualidade do dataset para o treinamento de modelos. De
fato, a técnica de aumento de dados pode evitar de forma significativa o
sobreajuste de modelos ao possibilitar que novas instâncias sejam
criadas artificialmente. Como observado no trabalho, os áudios criados
artificialmente se comportam de maneira muito próxima aos dados
originais, o que contribui para o treinamento do modelo de uma forma
geral.

Apesar da técnica de aumento de dados ter sido extremamente útil, não
ficou claro qual é o impacto significativo de cada técnica utilizada,
exceto pela adição de ruído, que se mostrou pouco útil no processo.
Outro fator importante que ficou pouco claro é a qualidade das fontes
dos dados, neste trabalho foram testadas duas fontes diferentes, e ambos
conjuntos de bases testadas tiveram um desempenho aceitável.

Trabalhos futuros
-----------------

Para trabalhos futuros sugere-se a realização de experimentos com
diferentes populações de dados, por exemplo, coleções de notícias do
*Youtube* (abordagem utilizada por [@bartz2017language]), gravações de
rádio e *podcasts*, e ainda, a realização de experimentos com a junção
destas bases. Como bem observado por [@revay2019multiclass], uma das
grandes dificuldades no desenvolvimento de sistemas LID é o treinamento
de modelos partindo de um único dataset, o modelo obtido pode se
confudir quando testado com áudios com características distintas das
utilizadas no momento do treinamento.

Outra sugestão é a realização de experimentos com relação a extração de
características além da obtenção de espectrogramas para o treinamento de
modelos e também a realização de experimentos com outros tipos de RNAs e
transferência de aprendizado.

[^1]: http://www.voxforge.org/

[^2]: O objetivo inicial do autor da pesquisa era o desenvolvimento de
    um sistema de reconhecimento automático de idiomas nos idiomas
    inglês, espanhol e português. Mais tarde, foram adicionados os
    idiomas alemão e francês em algumas das bases utilizadas.

[^3]: Algumas bases fornecem arquivos com a rotulação da fala dos
    discursos de cada áudio, isto é, o texto sendo falado naquele
    momento. Estes arquivos são úteis para o treinamento de modelos de
    transcrição automática de discurso.

[^4]: O nome CPD foi arbitrário, mas a motivação do nome é que *Corpus*
    é o nome dado as bases de discursos próprias para pesquisas na área
    de reconhecimento de fala.

[^5]: O GitHub é uma plataforma de hospedagem de código-fonte com
    controle de versão usando o Git e é uma ferramenta muito utilizada
    para projetos de software de qualquer tipo, incluindo códigos-fonte
    de classificadores e sistemas inteligentes.

[^6]: https://librivox.org/

[^7]: Atualmente o SLD pré-processado está disponível em
    https://www.kaggle.com/toponowicz/spoken-language-identification nos
    idiomas inglês, alemão e espanhol. Os áudios desta base apresentam
    características similares as bases utilizadas neste trabalho,
    entretanto, as bases utilizadas neste trabalho foram geradas por um
    algoritmo próprio.

[^8]: Mais tarde foi observado que uma duração maior seria ainda mais
    adequada devido a aplicação das técnicas de aumento de dados, já que
    elas alteram a duração dos áudios.

[^9]: https://jupyter.org/

[^10]: https://www.python.org/

[^11]: https://librosa.github.io/librosa/index.html

[^12]: http://sox.sourceforge.net/

[^13]: https://keras.io/

[^14]: Usualmente uma época é completada por iteração, mas pode-se
    modificar a quantidade de batches por iteração de forma que sejam
    necessários mais ou menos iterações para completar uma época.

[^15]: O número de épocas sugerido é bastante elevado porque neste
    experimento a ideia é interromper o treinamento após ter decorrido
    uma certa quantidade de tempo.

[^16]: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 16Gb de RAM e NVIDIA
    TITAN V.

[^17]: Em SLD a maioria dos audios possuíam uma taxa de amostragem de
    22050Hz enquanto em CPD a maioria dos áudios possuíam uma qualidade
    de 16000hz. Apesar disto, as bases cujos áudios foram convertidos
    para uma taxa inferior apresentaram desempenho similar as bases com
    taxas de amostragem mais altas.
