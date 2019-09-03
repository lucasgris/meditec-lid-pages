---
layout: post
title: Resultados gerais e pré-processamento do experimento 1 no conjunto de validação
categories: [Article,Results,Graphs,General,Preprocessing]
--- 

Com relação ao pré-processamento, o experimento 1 mostra que as diferentes técnicas testadas não produzem nenhum efeito significativo no treinamento. A principal diferença notada é o tempo necessário para treinar em alguns casos.

# Geral

O experimento 1 mostrou de uma forma geral que algumas bases convergiram muito mais rapidamente que outras. Algumas bases apresentaram uma certa convergência precoce já na primeira época, o que pode indicar problemas de sobreajuste, já que é possível que o número de filtros da topologia proposta seja superior ao adequado.

Na figura de cima é possível visualizar a evolução da acurácia de validação ao longo das épocas (o gráfico da direita está em escala logarítmica para facilitar a visualização), algumas bases apresentaram uma capacidade muito grande de aprendizado e outras demoraram um pouco mais para convergir. De uma forma geral, no entanto, todas as bases convergiram de uma forma ou de outra. 

![acurácia ao longo das épocas experimento 1](https://user-images.githubusercontent.com/34692520/64189847-a44be700-ce64-11e9-8125-7c1e190f878e.png)
![loss ao longo das épocas experimento 1](https://user-images.githubusercontent.com/34692520/64189864-ae6de580-ce64-11e9-9dc9-99c307637f14.png)

Se o resultado do teste final for positivo para todas as bases experimentadas, então é possível que a RNN tenha uma capacidade até surpreendente de lidar com diferentes características do áudio e abstrair conceitos de forma similar a um cérebro humano, do contrário, é possível que algumas das bases experimentadas sustentem vícios ao aprendizado do modelo e deveriam ser corrigidas de alguma forma.

# Técnicas de pré-processamento

As principais técnicas de pré-processamento utilizadas podem ser vistas na figura a seguir:

![Pre processamento gráfico](https://user-images.githubusercontent.com/34692520/64190233-669b8e00-ce65-11e9-8a35-edc61fbb7712.png)

No gráfico, a cor azul representa as bases com normalização diferentes da padrão, a cor amarela representa a utilização de filtro passa baixa e o triangulo representa a utilização de remoção de silêncio.

Ao observar o gráfico, nota-se que os efeitos alternativos de normalização testados parecem ter retardado um pouco o aprendizado. Outro fato observado é que os efeitos de remoção de silêncio e filtro passa baixa não produziram nenhum impacto significativo, mas isso pode mudar no conjunto de teste. É provável que estes efeitos de pré-processamento utilizados sejam até mesmo desnecessários. 

## Taxa de amostragem

Um fato interessante observado se deve a taxa de amostragem utilizada. Aparentemente a taxa de amostragem não proporcionou nenhum ganho ou nenhuma perda significativa quanto a acurácia do modelo, apenas quanto ao tempo necessário de treinamento (devido principalmente ao tamanho da imagem gerada). No gráfico seguinte, a cor cinza representa uma taxa de amostragem de 8khz, a cor vermelha uma taxa de amostragem de 16khz e a cor azul, uma taxa de 48khz.

![Taxa de amostragem gráfico](https://user-images.githubusercontent.com/34692520/64190252-73b87d00-ce65-11e9-92e1-add8487ac6cf.png)

Pode ser, no entanto, que se os audios originais tivessem mais qualidade, talvez a taxa de amostragem desempenhasse um papel importante.

# Conclusão

Aparentemente a rede neural consegue abstrair muito bem os efeitos de pré-processamento utilizados. Neste caso, pode ser interessante optar por efeitos que colaborem para diminuir o custo computacional do treinamento.
É claro que os resultados aqui mostrados se referem apenas ao conjunto de validação. Pode ser que o resultado no conjunto de teste seja totalmente diferente.
