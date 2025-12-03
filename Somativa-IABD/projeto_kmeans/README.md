
# Projeto - Agrupamento de Ações com K-Means (dados oficiais)

Este projeto utiliza o arquivo `dadosacoes.csv` fornecido pelo professor para:
- Fazer a exploração da base (`info`, `describe` e boxplots);
- Tratar a variável categórica com `get_dummies`;
- Aplicar o K-Means com 4, 5 e 8 clusters;
- Gerar os gráficos do método do cotovelo e da silhueta;
- Plotar gráficos 2D e 3D dos clusters.

## Como executar

1. Crie e ative um ambiente virtual (opcional, mas recomendado).
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Certifique-se de que o arquivo `dadosacoes.csv` esteja na mesma pasta do `main.py`.
4. Execute o script:

   ```bash
   python main.py
   ```

Os gráficos abrirão em janelas e, no terminal, você verá o resumo estatístico,
a inércia, silhueta e a quantidade de itens por cluster para k = 4, 5 e 8.
