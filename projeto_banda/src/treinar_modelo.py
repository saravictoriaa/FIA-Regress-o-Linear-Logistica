
from pathlib import Path

import joblib
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
    # 1. Localiza e carrega os dados preparados.
    pasta_projeto = Path(__file__).resolve().parents[1]
    pasta_dados = pasta_projeto / "data" / "processed"

    treino = pd.read_csv(pasta_dados / "treino.csv")
    teste = pd.read_csv(pasta_dados / "teste.csv")

    # Usamos somente as três medições anteriores como entradas.
    features = [
        "throughput_t_1",
        "throughput_t_2",
        "throughput_t_3",
    ]

    X_treino = treino[features]
    y_treino = treino["throughput_mbps"]

    X_teste = teste[features]
    y_teste = teste["throughput_mbps"]

    print("Dados carregados.")
    print(f"Registros de treino: {len(treino)}")
    print(f"Registros de teste: {len(teste)}")

    # 2. Cria e treina a regressão linear.
    modelo = LinearRegression()

    print("\nTreinando a regressão linear...")
    modelo.fit(X_treino, y_treino)

    # 3. Prevê os valores do conjunto de teste.
    previsoes = modelo.predict(X_teste)

    # 4. Calcula as métricas do modelo.
    mae = mean_absolute_error(y_teste, previsoes)
    rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
    r2 = r2_score(y_teste, previsoes)

    print("\n--- Métricas da regressão linear ---")
    print(f"MAE:  {mae:.4f} Mbps")
    print(f"RMSE: {rmse:.4f} Mbps")
    print(f"R²:   {r2:.4f}")

    # 5. Cria uma referência simples:
    # prever que a próxima medição será igual à última observada.
    previsao_referencia = X_teste["throughput_t_1"].to_numpy()

    mae_referencia = mean_absolute_error(
        y_teste, previsao_referencia
    )
    rmse_referencia = np.sqrt(
        mean_squared_error(y_teste, previsao_referencia)
    )

    print("\n--- Referência: repetir a última medição ---")
    print(f"MAE:  {mae_referencia:.4f} Mbps")
    print(f"RMSE: {rmse_referencia:.4f} Mbps")
    
    # 6. Compara valores reais e previstos ao longo do tempo.
    pasta_graficos = pasta_projeto / "results" / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    
    # Gráfico comparando os erros do modelo e da referência.
    figura, eixo = plt.subplots(figsize=(8, 5))

    metricas_nomes = ["MAE", "RMSE"]
    erros_modelo = [mae, rmse]
    erros_referencia = [mae_referencia, rmse_referencia]

    posicoes = np.arange(len(metricas_nomes))
    largura = 0.34

    barras_modelo = eixo.bar(
        posicoes - largura / 2,
        erros_modelo,
        largura,
        label="Regressão linear",
    )

    barras_referencia = eixo.bar(
        posicoes + largura / 2,
        erros_referencia,
        largura,
        label="Referência: última medição",
    )

    eixo.bar_label(barras_modelo, fmt="%.4f", padding=3, fontsize=9)
    eixo.bar_label(barras_referencia, fmt="%.4f", padding=3, fontsize=9)

    eixo.set_title("Comparação dos erros de previsão")
    eixo.set_ylabel("Erro (Mbps)")
    eixo.set_xticks(posicoes)
    eixo.set_xticklabels(metricas_nomes)
    eixo.grid(axis="y", linestyle="--", alpha=0.3)
    eixo.set_axisbelow(True)
    eixo.legend()

    figura.tight_layout()
    figura.savefig(
        pasta_graficos / "comparacao_metricas.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)


    figura, eixo = plt.subplots(figsize=(10, 5))

    eixo.plot(
        teste["tempo_alvo_segundos"],
        y_teste,
        label="Throughput real",
        linewidth=2,
    )

    eixo.plot(
        teste["tempo_alvo_segundos"],
        previsoes,
        label="Regressão linear",
        linewidth=1.8,
        linestyle="--",
    )

    eixo.plot(
        teste["tempo_alvo_segundos"],
        previsao_referencia,
        label="Referência: última medição",
        linewidth=1.5,
        linestyle=":",
    )

    eixo.set_title("Throughput real versus previsto")
    eixo.set_xlabel("Tempo (segundos)")
    eixo.set_ylabel("Throughput (Mbps)")
    eixo.grid(True, linestyle="--", alpha=0.3)
    eixo.legend()

    figura.tight_layout()
    figura.savefig(
        pasta_graficos / "comparacao_real_previsto.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figura)

    # 7. Salva o modelo treinado.
    pasta_modelos = pasta_projeto / "models"
    pasta_modelos.mkdir(parents=True, exist_ok=True)

    caminho_modelo = pasta_modelos / "regressao_linear.joblib"
    joblib.dump(modelo, caminho_modelo)

    print("\nGráfico salvo em:")
    print(pasta_graficos / "comparacao_real_previsto.png")

    print("\nModelo salvo em:")
    print(caminho_modelo)

    print("\nTreinamento e avaliação concluídos!")


if __name__ == "__main__":
    main()
