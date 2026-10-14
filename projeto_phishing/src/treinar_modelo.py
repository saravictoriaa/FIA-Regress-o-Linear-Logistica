
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # Localiza a pasta do projeto.
    pasta_projeto = Path(__file__).resolve().parents[1]
    pasta_dados = pasta_projeto / "data" / "processed"

    # Carrega os dados preparados na etapa anterior.
    treino = pd.read_csv(pasta_dados / "treino.csv")
    teste = pd.read_csv(pasta_dados / "teste.csv")

    # Separa as características (X) da resposta esperada (y).
    X_treino = treino.drop(columns=["result"])
    y_treino = treino["result"]

    X_teste = teste.drop(columns=["result"])
    y_teste = teste["result"]

    print("Dados carregados.")
    print(f"Registros de treino: {len(X_treino)}")
    print(f"Registros de teste: {len(X_teste)}")

    # Cria uma sequência de preparação e classificação.
    modelo = Pipeline([
        ("escalonamento", StandardScaler()),
        ("classificador", LogisticRegression(max_iter=1000)),
    ])

    # O modelo aprende usando apenas os dados de treino.
    print("\nTreinando a regressão logística...")
    modelo.fit(X_treino, y_treino)

    # Faz previsões nos dados reservados para o teste.
    previsoes = modelo.predict(X_teste)

    # Avalia o resultado.
    acuracia = accuracy_score(y_teste, previsoes)

    print(f"\nAcurácia: {acuracia:.4f}")
    print(f"Acurácia em porcentagem: {acuracia * 100:.2f}%")

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_teste,
            previsoes,
            labels=[-1, 1],
            target_names=["Phishing", "Legítimo"],
            zero_division=0,
        )
    )

    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=[-1, 1],
    )

    print("Matriz de confusão:")
    print("Linhas = classe real; colunas = classe prevista")
    print("Ordem das classes: Phishing, Legítimo")
    print(matriz)

    # Salva o modelo treinado para uso posterior.
    pasta_modelos = pasta_projeto / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_modelos / "regressao_logistica.joblib"
    joblib.dump(modelo, caminho_modelo)

    print(f"\nModelo salvo em: {caminho_modelo}")
    print("\nTreinamento concluído!")


if __name__ == "__main__":
    main()
