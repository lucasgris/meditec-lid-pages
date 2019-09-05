---
layout: post
title: Uma análise dos dados aumentados
categories: [Article,Results,Graphs,Augmented]
--- 

É realmente interessante como dados criados artificialmente podem contribuir para a qualidade do modelo. Algumas das técnicas de aumento de dados utilizadas foram muito promissoras.

# Data augmentation

Qual é o comportamento dos dados artificiais durante o treinamento?

Muito provavelmente os dados aumentados funcionariam de forma pior ou similar aos dados originais. É provável que, se os dados forem gerados corretamente, o comportamente seja muito próximo do real.
Como o aumento de dados aumenta o número de instâncias, a longo prazo o desempenho do modelo pode ultrapassar o desempenho utilizando somente dados reais. Aliás, é essa a ideia.

![grafico dados aumentados esperado](https://user-images.githubusercontent.com/34692520/64303098-211ab600-cf76-11e9-9e78-66fa2073f9a7.png)

## Resultados no experimento 1

Uma  das  tarefas  realizadas  que  realmente  melhorou  o  desempenho  dos  treinamentos foi a técnica de aumento de dados, ao observar a figura abaixo, percebe-se que os dados criados artificialmente pelo técnica parecem se comportar da mesma forma como os dados originais, possibilitando um poder maior de treinamento aos modelos. Nesta figura, a cor azul representa o uso de dados artificiais durante o treinamento e a cor cinza a ausência técnica, o triangulo representa as bases do conjunto CPD e o círculo as bases SLD. 

![aumento de dados experimento 1](https://user-images.githubusercontent.com/34692520/64303132-4a3b4680-cf76-11e9-85aa-3f70f2938a02.png)

Os treinamentos com as bases SLD tiveram um aumento muito consideravel, enquanto em CPD a melhora de desempenho foi mais sutil. Provavelmente porque a variedade de locutores em CPD é maior, então duas das técnicas de aumento de dados acabam não sendo tão importantes, isto é, a mudança de velocidade e de entonação.

Das técnicas de aumento de dados empregadas, a utilização de adicão de ruído pareceu fornecer pouca melhora no desempenho do treinamento.  Na figura a seguir, todas as bases mostradas apresentam aumento de dados, a cor cinza significa ausência de adicão de ruído.

![grafico adicao ruido](https://user-images.githubusercontent.com/34692520/64303204-938b9600-cf76-11e9-84c4-102965c0cd52.png)

## Resultados no experimento 2

No experimento 2 a técnica de aumento de dados ́e colocada em desvantagem. Como o experimento foi realizado de forma que cada iteração e realizada com a mesma quantidade de dados para todos os treinamentos, a maior proporção dos dados das basescom aumento de dados ́e artificial. Espera-se portanto um comportamento pior ou similar ao real. De fato, o processo de aumento de dados funcionou como o esperado, conforme pode ser visto na figura seguinte, o desempenho das bases com dados artificiais ficou muito pr ́oximo do desempenho das bases contendo apenas dados originais.

![aumento de dados experimento 2](https://user-images.githubusercontent.com/34692520/64303167-6212ca80-cf76-11e9-9207-3ba06a9aafc2.png)

_Compare este gráfico com a imagem mostrada lá em cima_

# Conclusão

A técnica de aumento de dados é muito importante e muito útil. Se for feita adequadamente, pode aumentar a capacidade do modelo de aprender e ainda combater o overfitting.
