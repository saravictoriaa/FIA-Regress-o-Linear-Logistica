
from pathlib import Path

import joblib
import pandas as pd


def main():
    pasta_projeto = Path(__file__).resolve().parents[1]

    caminho_modelo = pasta_projeto / "models" / "regressao_logistica.joblib"
    caminho_teste = (
        pasta_projeto / "data" / "processed" / "teste.csv"
    )

    # Verifica se os arquivos necessários existem.
    if not caminho_modelo.exists():
        print("Modelo não encontrado. Execute treinar_modelo.py primeiro.")
        return

    if not caminho_teste.exists():
        print("Arquivo de teste não encontrado. Prepare os dados primeiro.")
        return

    # Carrega o modelo treinado e os dados de teste.
    modelo = joblib.load(caminho_modelo)
    dados = pd.read_csv(caminho_teste)

    X_teste = dados.drop(columns=["result"])

    print("Detector de phishing")
    print(f"Registros disponíveis: {len(dados)}")
    print("Digite um índice entre 0 e", len(dados) - 1)

    try:
        indice = int(input("Índice do registro para analisar: "))
    except ValueError:
        print("Erro: digite um número inteiro.")
        return

    if indice < 0 or indice >= len(dados):
        print("Índice inválido.")
        return

    # Seleciona um registro, mantendo o formato de tabela esperado.
    registro = X_teste.iloc[[indice]]

    # Faz a classificação.
    previsao = int(modelo.predict(registro)[0])

    # Obtém as probabilidades estimadas para cada classe.
    probabilidades = modelo.predict_proba(registro)[0]
    classes = modelo.named_steps["classificador"].classes_

    nomes = {
        -1: "Phishing",
        1: "Legítimo"
    }

    print("\n--- Resultado ---")
    print(f"Classe prevista: {nomes[previsao]} ({previsao})")

    print("\nProbabilidades estimadas:")
    for classe, probabilidade in zip(classes, probabilidades):
        print(
            f"{nomes[int(classe)]}: "
            f"{probabilidade * 100:.2f}%"
        )


    classe_real = int(dados.iloc[indice]["result"])

    print(f"\nClasse real na base: {nomes[classe_real]}")
    print(
        "A previsão coincidiu com a classe real."
        if previsao == classe_real
        else "A previsão não coincidiu com a classe real."
    )


if __name__ == "__main__":
    main()
