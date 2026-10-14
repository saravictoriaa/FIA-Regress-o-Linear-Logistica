
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedGroupKFold


def main():
    print("Carregando a base de phishing...")

    base = fetch_ucirepo(id=327)

    X = base.data.features.copy()
    y = base.data.targets["result"].copy()

    # Agrupa registros com exatamente as mesmas características.
    # Assim, um mesmo perfil não será dividido entre treino e teste.
    grupos = pd.MultiIndex.from_frame(X).factorize()[0]

    # Cria cinco partes, tentando preservar a proporção das classes.
    #Uma delas sebndo usada como teste e as outras como treino.
    separador = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    indices_treino, indices_teste = next(
        separador.split(X, y, groups=grupos)
    )

    # Separa as características e acrescenta a classe ao resultado.
    treino = X.iloc[indices_treino].copy()
    treino["result"] = y.iloc[indices_treino].to_numpy()

    teste = X.iloc[indices_teste].copy()
    teste["result"] = y.iloc[indices_teste].to_numpy()

    # Confere se algum perfil de características aparece nos dois conjuntos.
    perfis_treino = pd.MultiIndex.from_frame(
        X.iloc[indices_treino]
    ).unique()

    perfis_teste = pd.MultiIndex.from_frame(
        X.iloc[indices_teste]
    ).unique()

    perfis_compartilhados = perfis_treino.intersection(
        perfis_teste
    )

    print("\n--- Resultado da preparação ---")
    print(f"Registros originais: {len(X)}")
    print(f"Registros para treino: {len(treino)}")
    print(f"Registros para teste: {len(teste)}")

    print("\nDistribuição das classes no treino (%):")
    print(
        (
            treino["result"].value_counts(normalize=True).sort_index()
            * 100
        ).round(1)
    )

    print("\nDistribuição das classes no teste (%):")
    print(
        (
            teste["result"].value_counts(normalize=True).sort_index()
            * 100
        ).round(1)
    )

    print(
        "\nPerfis de características presentes nos dois conjuntos:",
        len(perfis_compartilhados)
    )

    if len(perfis_compartilhados) > 0:
        raise RuntimeError(
            "Foram encontrados perfis compartilhados entre treino e teste."
        )

    # Salva os conjuntos preparados.
    pasta_saida = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed"
    )
    pasta_saida.mkdir(parents=True, exist_ok=True)

    treino.to_csv(pasta_saida / "treino.csv", index=False)
    teste.to_csv(pasta_saida / "teste.csv", index=False)

    print("\nArquivos salvos em:")
    print(pasta_saida)
    print("\nPreparação concluída!")


if __name__ == "__main__":
    main()
