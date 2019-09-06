---
layout: post
title: Resultados finais: até que ponto podemos confiar em um modelo de rede neural?
categories: [Article,Results,Graphs,Augmented]
--- 

Os testes finais contra os datasets de teste fornecerem informações contraditórias. De um lado, os modelos obtidos com a base CPD obtiveram um desempenho muito bom no conjunto de teste CPD, mas um desempenho ruim no conjunto SLD. Os modelos obtidos com SLd obtiveram resultados razoáveis, entretanto similares em ambas as bases. Isso sugere que a base CPD é mais suscetível a vícios. 

# Bases utilizadas

As bases utilizadas para a realização dos experimentos foram:
- CPD (_Corpus Dataset_): Base construída a partir de diversos corpus obtidos na Web como a [VoxForge](http://www.voxforge.org/), a [Ciempiess](http://www.ciempiess.org) e a [LibriSpeech](http://www.openslr.org/12).
- SLD (_Spoken Language Dataset_): Base construída a aprtir de gravações LibriVox. Similar a [este projeto](https://github.com/tomasz-oponowicz/spoken_language_dataset/tree/a4235d4710377607f20a9b967ce3447e0c4a11a4).

# Resultados 

Primeiro foram testados todos os modelos obtidos com seus respectivos conjuntos de teste. Os conjuntos de teste não foram feitos com aumento de dados, mas tiveram as mesmas características de pré-processamento aplicadas nos demais conjuntos de treinamento e validação.

O resultado pode ser visto na tabela seguinte:

![tabela teste final](https://user-images.githubusercontent.com/34692520/64397357-8642da00-d050-11e9-9684-a62cdbc1fe3b.png)

Selecionando alguns modelos promissores, foram realizados alguns testes novamente, mas desta vez com outro dataset de teste de uma população diferente. Na verdade, testou-se os modelos CPD om os datasets SLD e vice-versa. O resultado pode ser visto a seguir:

![tabela teste final com bases opostas](https://user-images.githubusercontent.com/34692520/64397310-5e537680-d050-11e9-977e-889f18d818c0.png)

## O problema do vício: quando a rede aprende que o Lobo é a neve branca!

Nem sempre a acurácia alta significa que o modelo realmente aprendeu as características que era desejáveis inicialmente. Por se tratar de um algoritmo caixa preta, pode ser complicado entender o que a rede realmente aprendeu.

Em particular, um dataset ruim sempre produzirá resultados ruins!

### [Why Should I Trust You?](https://arxiv.org/abs/1602.04938)

![A rede aprendeu certo do jeito errado](https://hackernoon.com/hn-images/1*H6w9DUlhLSoA6-CSwLpO5Q.png)

[Dogs, Wolves, Data Science, and Why Machines Must Learn Like Humans Do](https://hackernoon.com/dogs-wolves-data-science-and-why-machines-must-learn-like-humans-do-41c43bc7f982)

# Conclusão

O dataset CPD propiciou modelos com acurácias muito boas em um primeiro momento. Uma análise com mais cuidado revelou que na verdade nào se pode confiar nos modelos obtidos utilizando somente esse dataset, porque ele aparentemente sustenta a criação de vícios no processo de aprendizagem.
