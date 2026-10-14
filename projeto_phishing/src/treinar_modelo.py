
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Localiza e carrega os dados preparados.
    pasta_projeto = Path(__file__).resolve().parents[1]
    pasta_dados = pasta_projeto / "data" / "processed"

    treino = pd.read_csv(pasta_dados / "treino.csv")
    teste = pd.read_csv(pasta_dados / "teste.csv")

    X_treino = treino.drop(columns=["result"])
    y_treino = treino["result"]

    X_teste = teste.drop(columns=["result"])
    y_teste = teste["result"]

    print("Dados carregados.")
    print(f"Registros de treino: {len(X_treino)}")
    print(f"Registros de teste: {len(X_teste)}")

    # 2. Cria e treina o modelo.
    modelo = Pipeline([
        ("escalonamento", StandardScaler()),
        ("classificador", LogisticRegression(max_iter=1000)),
    ])

    print("\nTreinando a regressão logística...")
    modelo.fit(X_treino, y_treino)

    # 3. Faz previsões no conjunto de teste.
    previsoes = modelo.predict(X_teste)

    acuracia = accuracy_score(y_teste, previsoes)

    print(f"\nAcurácia: {acuracia:.4f}")
    print(f"Acurácia em porcentagem: {acuracia * 100:.2f}%")

    nomes_classes = ["Phishing", "Legítimo"]
    ordem_classes = [-1, 1]

    # Guarda as métricas para exibição e geração do gráfico.
    relatorio = classification_report(
        y_teste,
        previsoes,
        labels=ordem_classes,
        target_names=nomes_classes,
        zero_division=0,
        output_dict=True,
    )

    print("\nRelatório de classificação:")
    print(
        classification_report(
            y_teste,
            previsoes,
            labels=ordem_classes,
            target_names=nomes_classes,
            zero_division=0,
        )
    )

    # Cria a pasta de saída dos gráficos.
    pasta_graficos = pasta_projeto / "results" / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    # 4. Gráfico da matriz de confusão.
    matriz = confusion_matrix(
        y_teste,
        previsoes,
        labels=ordem_classes,
    )

    figura, eixo = plt.subplots(figsize=(7, 5))

    exibicao = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nomes_classes,
    )

    exibicao.plot(
        ax=eixo,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    eixo.set_title("Matriz de confusão — Regressão logística")
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")

    figura.tight_layout()
    figura.savefig(
        pasta_graficos / "matriz_confusao.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)

    # 5. Gráfico de precision, recall e F1-score por classe.
    metricas = ["precision", "recall", "f1-score"]
    posicoes = np.arange(len(nomes_classes))
    largura = 0.24

    figura, eixo = plt.subplots(figsize=(9, 5))

    for i, metrica in enumerate(metricas):
        valores = [
            relatorio[nome][metrica]
            for nome in nomes_classes
        ]

        barras = eixo.bar(
            posicoes + (i - 1) * largura,
            valores,
            largura,
            label=metrica.capitalize(),
        )

        eixo.bar_label(
            barras,
            fmt="%.2f",
            padding=2,
            fontsize=8,
        )

    eixo.set_title("Métricas de classificação por classe")
    eixo.set_ylabel("Pontuação")
    eixo.set_xticks(posicoes)
    eixo.set_xticklabels(nomes_classes)
    eixo.set_ylim(0, 1.12)
    eixo.grid(axis="y", linestyle="--", alpha=0.3)
    eixo.set_axisbelow(True)
    eixo.legend()

    figura.tight_layout()
    figura.savefig(
        pasta_graficos / "metricas_classificacao.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)

    # 6. Curva ROC, considerando phishing (-1) como classe positiva.
    y_teste_phishing = (y_teste == -1).astype(int)

    classes_modelo = list(
        modelo.named_steps["classificador"].classes_
    )
    indice_phishing = classes_modelo.index(-1)

    probabilidades_phishing = modelo.predict_proba(X_teste)[
        :, indice_phishing
    ]

    fpr, tpr, _ = roc_curve(
        y_teste_phishing,
        probabilidades_phishing,
    )

    auc = roc_auc_score(
        y_teste_phishing,
        probabilidades_phishing,
    )

    print(f"AUC-ROC (phishing como classe positiva): {auc:.4f}")

    figura, eixo = plt.subplots(figsize=(7, 5))

    eixo.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Regressão logística (AUC = {auc:.3f})",
    )

    eixo.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Referência aleatória",
    )

    eixo.set_title("Curva ROC — Detecção de phishing")
    eixo.set_xlabel("Taxa de falsos positivos")
    eixo.set_ylabel("Taxa de verdadeiros positivos")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1.02)
    eixo.grid(linestyle="--", alpha=0.3)
    eixo.legend(loc="lower right")

    figura.tight_layout()
    figura.savefig(
        pasta_graficos / "curva_roc.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)

    # 7. Salva o modelo treinado.
    pasta_modelos = pasta_projeto / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_modelos / "regressao_logistica.joblib"
    joblib.dump(modelo, caminho_modelo)

    print("\nGráficos salvos em:")
    print(pasta_graficos)

    print("\nModelo salvo em:")
    print(caminho_modelo)

    print("\nTreinamento, avaliação e geração dos gráficos concluídos!")


if __name__ == "__main__":
    main()
