# Agrupamento de Ações com K-Means

Este repositório contém a implementação de um estudo de agrupamento de ações utilizando o algoritmo K-Means, como parte da avaliação da disciplina de Inteligência Artificial e Big Data.

A partir de uma base de dados contendo informações de diferentes ações (nome, preço, quantidade de cotas e valor de mercado), são aplicadas técnicas de aprendizado não supervisionado para identificar grupos (clusters) de ativos com características semelhantes.

---

## Objetivos do projeto

- Carregar e preparar a base de dados oficial fornecida pelo professor (`dadosacoes.csv`).
- Realizar exploração estatística dos dados (informações gerais e descrição numérica).
- Tratar a variável categórica com codificação one-hot (`get_dummies`).
- Aplicar o algoritmo K-Means com diferentes valores de `k` (4, 5 e 8 clusters).
- Avaliar os resultados por meio:
  - da inércia (método do cotovelo);
  - do coeficiente de silhueta.
- Visualizar os clusters em gráficos 2D e 3D para análise qualitativa.

---

## Tecnologias utilizadas

- Python 3.x  
- Bibliotecas:
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

Todas as dependências necessárias estão listadas em `requirements.txt`.

---

## Estrutura do projeto

A estrutura básica do repositório é:

```text
projeto_kmeans_dadosacoes_v2/
├── main.py
├── dadosacoes.csv
├── requirements.txt
└── README.md
