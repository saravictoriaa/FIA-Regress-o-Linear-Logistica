
from pathlib import Path
import math

import joblib
import pandas as pd


def main():
    # Localiza o projeto e o modelo treinado.
    pasta_projeto = Path(__file__).resolve().parents[1]
    caminho_modelo = (
        pasta_projeto / "models" / "regressao_linear.joblib"
    )

    if not caminho_modelo.exists():
        print("Modelo não encontrado.")
        print("Execute treinar_modelo.py antes de fazer previsões.")
        return

    # Carrega o modelo salvo, sem realizar um novo treinamento.
    modelo = joblib.load(caminho_modelo)

    print("PREVISÃO DE THROUGHPUT DA REDE")
    print("-" * 35)
    print("O modelo utiliza a última medição conhecida como entrada.")

    try:
        entrada_usuario = input(
            "\nInforme a última vazão medida (Mbps): "
        ).strip()

        # Permite digitar números com ponto ou vírgula decimal.
        throughput_atual = float(
            entrada_usuario.replace(",", ".")
        )

    except ValueError:
        print("Entrada inválida. Informe um número, como 3.5.")
        return

    if not math.isfinite(throughput_atual):
        print("Informe um valor numérico finito.")
        return

    if throughput_atual < 0:
        print("A vazão não pode ser negativa.")
        return

    # Usa o mesmo nome de variável empregado no treinamento.
    entrada = pd.DataFrame({
        "throughput_t_1": [throughput_atual]
    })

    previsao = float(modelo.predict(entrada)[0])

    print("\n--- Resultado ---")
    print(f"Última medição informada: {throughput_atual:.4f} Mbps")
    print(f"Throughput previsto: {previsao:.4f} Mbps")

    # Referência simples usada na avaliação do projeto.
    print(
        "Referência (repetir a última medição): "
        f"{throughput_atual:.4f} Mbps"
    )

    if previsao < 0:
        print(
            "\nAtenção: a regressão linear produziu um valor negativo. "
            "Isso pode ocorrer porque o modelo não restringe suas "
            "previsões a valores fisicamente possíveis."
        )

    print(
        "\nA previsão representa a próxima medição da série, "
        "não um instante futuro com duração fixa."
    )
    print(
        "O resultado é uma estimativa, não uma garantia "
        "do throughput que será observado."
    )


if __name__ == "__main__":
    main()
