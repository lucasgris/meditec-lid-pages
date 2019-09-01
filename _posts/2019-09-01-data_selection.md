---
layout: post
title: Seleção e análise dos dados
categories: [Article,Data]
--- 
A primeira etapa no processo de desenvolvimento do sistema consistiu na seleção e análise de arquivos de áudio para a geração das bases experimentais, o pré-processamento dos dados e a realização dos experimentos. Inicialmente procurou-se a obtenção de uma base própria para este fim, ou uma base de discursos que contivesse todos os idiomas alvo para o treinamento dos modelos. O objetivo inicial era o desenvolvimento de um sistema de reconhecimento automático de idiomas nos idiomas inglês, espanhol e português. Mais tarde, foram adicionados os idiomas alemão e francês em algumas das bases utilizadas.

Nesta parte foram realizadas pesquisas de possíveis áudios adequados ao problema e em seguida foram realizadas análises desses áudios para verificar quais eram as características originais dos dados obtidos, isto é, a duração total das instâncias, a taxa de amostragem original, a quantidade de instância de cada idioma, a presença ou não de ruídos, a quantidade de locutores e a quantidade de locutores dos gêneros masculino e feminino.

As análises estão disponíveis em [https://github.com/lucasgris/meditec-lid/tree/master/data/data_analysis](https://github.com/lucasgris/meditec-lid/tree/master/data/data_analysis)
