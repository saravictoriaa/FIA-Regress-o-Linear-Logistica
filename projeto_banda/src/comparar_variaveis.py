
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def main():
    pasta_projeto = Path(__file__).resolve().parents[1]
    pasta_dados = pasta_projeto / "data" / "processed"

    # Usamos somente o conjunto de treino para comparar configurações.
    dados = pd.read_csv(pasta_dados / "treino.csv")

    # Divisão cronológica interna: primeiros 80% para ajustar
    # os modelos e últimos 20% para validar as configurações.
    ponto_corte = int(len(dados) * 0.8)

    treino_interno = dados.iloc[:ponto_corte]
    validacao = dados.iloc[ponto_corte:]

    alvo_treino = treino_interno["throughput_mbps"]
    alvo_validacao = validacao["throughput_mbps"]

    configuracoes = {
        "1 medição anterior": ["throughput_t_1"],
        "2 medições anteriores": [
            "throughput_t_1",
            "throughput_t_2",
        ],
        "3 medições anteriores": [
            "throughput_t_1",
            "throughput_t_2",
            "throughput_t_3",
        ],
    }

    resultados = []

    print("Comparando configurações de regressão linear...")
    print(f"Registros para ajuste: {len(treino_interno)}")
    print(f"Registros para validação: {len(validacao)}")

    # Testa cada configuração usando exatamente os mesmos
    # períodos de ajuste e validação.
    for nome, features in configuracoes.items():
        modelo = LinearRegression()

        modelo.fit(
            treino_interno[features],
            alvo_treino,
        )

        previsoes = modelo.predict(validacao[features])

        resultados.append({
            "Configuracao": nome,
            "MAE": mean_absolute_error(
                alvo_validacao, previsoes
            ),
            "RMSE": np.sqrt(
                mean_squared_error(alvo_validacao, previsoes)
            ),
            "R2": r2_score(alvo_validacao, previsoes),
        })

    # Referência: repetir a última medição conhecida.
    previsao_referencia = validacao["throughput_t_1"]

    resultados.append({
        "Configuracao": "Referência: última medição",
        "MAE": mean_absolute_error(
            alvo_validacao, previsao_referencia
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                alvo_validacao, previsao_referencia
            )
        ),
        "R2": r2_score(
            alvo_validacao, previsao_referencia
        ),
    })

    tabela = pd.DataFrame(resultados).sort_values("MAE")

    print("\n--- Resultados da validação ---")
    print(tabela.round(4).to_string(index=False))

    # Gráfico comparativo de MAE.
    pasta_graficos = pasta_projeto / "results" / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    figura, eixo = plt.subplots(figsize=(10, 5))

    barras = eixo.bar(
        tabela["Configuracao"],
        tabela["MAE"],
    )

    eixo.bar_label(barras, fmt="%.4f", padding=3)
    eixo.set_title("Comparação das configurações — validação")
    eixo.set_ylabel("MAE (Mbps)")
    eixo.set_xlabel("Configuração")
    eixo.grid(axis="y", linestyle="--", alpha=0.3)
    eixo.set_axisbelow(True)

    plt.xticks(rotation=15, ha="right")
    figura.tight_layout()

    caminho_grafico = pasta_graficos / "comparacao_variaveis.png"
    figura.savefig(
        caminho_grafico,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)

    print("\nGráfico salvo em:")
    print(caminho_grafico)

    print(
        "\nUse os resultados de validação para escolher a configuração."
    )


if __name__ == "__main__":
    main()
