
from ucimlrepo import fetch_ucirepo


def main():
    print("Carregando a base de phishing...")

    # Busca o conjunto de dados oficial da UCI.
    base = fetch_ucirepo(id=327)

    # Separação de características e  variável que queremos prever.
    X = base.data.features
    y = base.data.targets

    print("\nNome da base:")
    print(base.metadata.name)

    print("\nDimensões das características (X):")
    print(X.shape)

    print("\nDimensões do alvo (y):")
    print(y.shape)

    print("\nPrimeiras cinco linhas das características:")
    print(X.head())

    print("\nPrimeiras cinco linhas do alvo:")
    print(y.head())

    print("\nValores ausentes em X:")
    print(X.isna().sum().sum())

    print("\nValores ausentes em y:")
    print(y.isna().sum().sum())

    
    print("\nDistribuição das classes:")
    contagem = y["result"].value_counts().sort_index()

    total = len(y)
    nomes_classes = {
        -1: "Phishing",
        1: "Legítimo"
    }

    for classe, quantidade in contagem.items():
        nome = nomes_classes.get(int(classe), "Classe desconhecida")
        percentual = quantidade / total * 100

        print(
            f"{nome} ({classe}): "
            f"{quantidade} registros ({percentual:.1f}%)"
        )

    
    # Diagnóstico de registros duplicados
    dados_completos = X.join(y)

    duplicados_completos = dados_completos.duplicated().sum()
    combinacoes_unicas = len(dados_completos.drop_duplicates())
    perfis_unicos = len(X.drop_duplicates())

    print("\n--- Diagnóstico de duplicatas ---")
    print(f"Total de registros: {len(dados_completos)}")
    print(f"Ocorrências repetidas além da primeira: {duplicados_completos}")
    print(f"Combinações únicas de características e classe: {combinacoes_unicas}")
    print(f"Combinações únicas de características: {perfis_unicos}")

    # Verifica se o mesmo perfil de características aparece
    # associado a mais de uma classe.
    classes_por_perfil = (
        dados_completos
        .groupby(list(X.columns), dropna=False)["result"]
        .nunique()
    )

    perfis_conflitantes = int((classes_por_perfil > 1).sum())

    print(
        "Perfis de características associados a classes diferentes:",
        perfis_conflitantes
    )

if __name__ == "__main__":
    main()
