
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def carregar_dados(caminho_csv: str):
    """
    Lê o CSV oficial `dadosacoes.csv` e renomeia as colunas para nomes
    simples que o restante do código usa.

    Colunas esperadas no arquivo original:
    - 'nome ação'
    - 'preço ação R$'
    - 'qtde cotas'
    - 'valor de mercado R$ -(Bilhões)'
    """
    df = pd.read_csv(caminho_csv)

    # Mapeamento direto das colunas originais -> nomes usados no código
    rename_map = {
        "nome ação": "nome_acao",
        "preço ação R$": "preco_acao",
        "qtde cotas": "qtde_cotas",
        "valor de mercado R$ -(Bilhões)": "valor_mercado_bilhoes",
    }

    # Para evitar problema de maiúscula/minúscula/espaco, normalizamos a chave
    colunas_orig = {c.strip(): c for c in df.columns}
    efetivo_rename = {}
    for chave_original, novo_nome in rename_map.items():
        # procura a coluna que corresponde a essa chave
        for col in df.columns:
            if col.strip().lower() == chave_original.lower():
                efetivo_rename[col] = novo_nome

    df = df.rename(columns=efetivo_rename)

    # Só para conferência: garante que as 4 colunas existem
    colunas_esperadas = ["nome_acao", "preco_acao", "qtde_cotas", "valor_mercado_bilhoes"]
    for col in colunas_esperadas:
        if col not in df.columns:
            raise ValueError(f"Coluna esperada não encontrada após renomear: {col}. "
                             f"Verifique se o CSV é o oficial do professor.")

    # df_original aqui é igual ao lido, antes de mexer nos tipos, etc.
    df_original = df.copy()

    return df, df_original

def exploracao_dados(df: pd.DataFrame) -> None:
    print("\n================ INFO DO DATAFRAME ================")
    print(df.info())

    print("\n================ DESCRICAO ESTATISTICA ================")
    print(df.describe())

    # Boxplot do preço da ação por ativo
    plt.figure()
    sns.boxplot(data=df, x="nome_acao", y="preco_acao")
    plt.title("Boxplot do Preço da Ação por Ativo")
    plt.xlabel("Nome da ação")
    plt.ylabel("Preço da ação (R$)")
    plt.tight_layout()
    plt.show()

    # Boxplot do valor de mercado por ativo
    plt.figure()
    sns.boxplot(data=df, x="nome_acao", y="valor_mercado_bilhoes")
    plt.title("Boxplot do Valor de Mercado por Ativo")
    plt.xlabel("Nome da ação")
    plt.ylabel("Valor de mercado (bilhões de R$)")
    plt.tight_layout()
    plt.show()

def pre_processamento(df: pd.DataFrame) -> pd.DataFrame:
    # Trata valores ausentes, se existirem
    df = df.copy()
    df = df.dropna()

    # Converte variável categórica em dummies (one-hot)
    df_dummies = pd.get_dummies(df, columns=["nome_acao"], drop_first=True)

    return df_dummies

def aplicar_kmeans(df_proc: pd.DataFrame, n_clusters: int):
    # Seleciona apenas colunas numéricas para o K-means
    colunas_numericas = df_proc.select_dtypes(include="number").columns.tolist()

    # Escalonamento (normalização) dos dados
    scaler = StandardScaler()
    dados_escalonados = scaler.fit_transform(df_proc[colunas_numericas])

    modelo = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = modelo.fit_predict(dados_escalonados)

    df_clusters = df_proc.copy()
    df_clusters["cluster"] = labels

    # Inércia e silhueta
    inercia = modelo.inertia_
    silhueta = None
    if n_clusters > 1:
        silhueta = silhouette_score(dados_escalonados, labels)

    return df_clusters, modelo, inercia, silhueta, colunas_numericas, dados_escalonados

def graficos_kmeans(dados_escalonados, max_k: int = 8):
    # Gráfico do cotovelo
    inercia_por_k = []
    ks = list(range(1, max_k + 1))
    for k in ks:
        modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
        modelo.fit(dados_escalonados)
        inercia_por_k.append(modelo.inertia_)

    plt.figure()
    plt.plot(ks, inercia_por_k, marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia (Soma dos quadrados)")
    plt.title("Método do Cotovelo")
    plt.tight_layout()
    plt.show()

    # Gráfico da silhueta (para k >= 2)
    silhuetas = []
    ks_sil = list(range(2, max_k + 1))
    for k in ks_sil:
        modelo = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = modelo.fit_predict(dados_escalonados)
        score = silhouette_score(dados_escalonados, labels)
        silhuetas.append(score)

    plt.figure()
    plt.plot(ks_sil, silhuetas, marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Coeficiente de silhueta")
    plt.title("Análise de Silhueta")
    plt.tight_layout()
    plt.show()

def visualizar_clusters_2d(df_original_limpo: pd.DataFrame, df_clusters: pd.DataFrame):
    # Junta novamente com as colunas originais para facilitar visualização
    df_plot = df_original_limpo.join(df_clusters["cluster"])

    plt.figure()
    sns.scatterplot(
        data=df_plot,
        x="preco_acao",
        y="valor_mercado_bilhoes",
        hue="cluster",
    )
    plt.xlabel("Preço da ação (R$)")
    plt.ylabel("Valor de mercado (bilhões de R$)")
    plt.title("Clusters em 2D - Preço x Valor de mercado")
    plt.tight_layout()
    plt.show()

def visualizar_clusters_3d(df_original_limpo: pd.DataFrame, df_clusters: pd.DataFrame):
    df_plot = df_original_limpo.join(df_clusters["cluster"])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df_plot["preco_acao"],
        df_plot["qtde_cotas"],
        df_plot["valor_mercado_bilhoes"],
        c=df_plot["cluster"],
    )
    ax.set_xlabel("Preço da ação (R$)")
    ax.set_ylabel("Quantidade de cotas")
    ax.set_zlabel("Valor de mercado (bilhões de R$)")
    ax.set_title("Clusters em 3D")
    plt.tight_layout()
    plt.show()

def main():
    caminho_csv = "dadosacoes.csv"  # arquivo oficial enviado pelo professor
    df, df_original = carregar_dados(caminho_csv)

    # Garante que estamos trabalhando apenas com as colunas principais
    df_base = df[["nome_acao", "preco_acao", "qtde_cotas", "valor_mercado_bilhoes"]].copy()

    exploracao_dados(df_base)

    df_proc = pre_processamento(df_base)

    # K-means com 4 clusters
    print("\n================ K-MEANS COM 4 CLUSTERS ================")
    df_4, modelo_4, inercia_4, sil_4, colunas_numericas, dados_escalonados = aplicar_kmeans(df_proc, 4)
    print("Inércia (k=4):", inercia_4)
    print("Silhueta (k=4):", sil_4)
    print("Quantidade de itens por cluster (k=4):")
    print(df_4["cluster"].value_counts())

    # K-means com 5 clusters
    print("\n================ K-MEANS COM 5 CLUSTERS ================")
    df_5, modelo_5, inercia_5, sil_5, _, _ = aplicar_kmeans(df_proc, 5)
    print("Inércia (k=5):", inercia_5)
    print("Silhueta (k=5):", sil_5)
    print("Quantidade de itens por cluster (k=5):")
    print(df_5["cluster"].value_counts())

    # K-means com 8 clusters
    print("\n================ K-MEANS COM 8 CLUSTERS ================")
    df_8, modelo_8, inercia_8, sil_8, _, _ = aplicar_kmeans(df_proc, 8)
    print("Inércia (k=8):", inercia_8)
    print("Silhueta (k=8):", sil_8)
    print("Quantidade de itens por cluster (k=8):")
    print(df_8["cluster"].value_counts())

    # Gráficos do cotovelo e da silhueta
    graficos_kmeans(dados_escalonados, max_k=8)

    # Visualizações 2D e 3D usando o modelo com 4 clusters
    visualizar_clusters_2d(df_base, df_4)
    visualizar_clusters_3d(df_base, df_4)

    print("\nA maior vantagem do aprendizado não supervisionado em relação ao supervisionado")
    print("é que ele consegue descobrir automaticamente padrões e agrupamentos escondidos nos dados,")
    print("mesmo quando não temos rótulos prontos para cada exemplo.")

if __name__ == "__main__":
    main()
