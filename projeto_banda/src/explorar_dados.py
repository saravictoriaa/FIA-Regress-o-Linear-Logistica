from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Localiza as pastas do projeto.
    pasta_projeto = Path(__file__).resolve().parents[1]

    caminho_dados = (
        pasta_projeto / "data" / "raw" / "norway_bus_1.txt"
    )

    pasta_graficos = pasta_projeto / "results" / "graficos"
    pasta_graficos.mkdir(parents=True, exist_ok=True)

    print("Carregando as medições de banda...")

    # O arquivo contém duas colunas, separadas por espaços,
    # e não possui cabeçalho.
    dados = pd.read_csv(
        caminho_dados,
        sep=r"\s+",
        header=None,
        names=["tempo_segundos", "throughput_mbps"],
    )

    # 1. Dimensões e primeiras linhas.
    print("\n--- Estrutura dos dados ---")
    print(f"Quantidade de registros: {len(dados)}")
    print(f"Quantidade de colunas: {len(dados.columns)}")

    print("\nPrimeiras cinco medições:")
    print(dados.head())

    print("\nÚltimas cinco medições:")
    print(dados.tail())

    print("\nTipos das colunas:")
    print(dados.dtypes)

    # 2. Verificação da qualidade dos dados.
    print("\n--- Qualidade dos dados ---")
    print("\nValores ausentes:")
    print(dados.isna().sum())

    print(
        "\nLinhas duplicadas:",
        dados.duplicated().sum(),
    )

    tempos = dados["tempo_segundos"]

    print(
        "\nOs tempos estão em ordem crescente?",
        tempos.is_monotonic_increasing,
    )

    # 3. Estatísticas da vazão.
    print("\n--- Estatísticas da vazão (Mbps) ---")
    print(dados["throughput_mbps"].describe())

    # 4. Analisa os intervalos entre as medições.
    intervalos = tempos.diff().dropna()

    print("\n--- Intervalos entre medições ---")
    print(intervalos.describe())

    # 5. Gera um gráfico das medições ao longo do tempo.
    figura, eixo = plt.subplots(figsize=(10, 5))

    eixo.plot(
        dados["tempo_segundos"],
        dados["throughput_mbps"],
        linewidth=1.2,
    )

    eixo.set_title("Vazão da rede ao longo do tempo")
    eixo.set_xlabel("Tempo (segundos)")
    eixo.set_ylabel("Vazão (Mbps)")
    eixo.grid(True, linestyle="--", alpha=0.3)

    figura.tight_layout()

    caminho_grafico = (
        pasta_graficos / "vazao_ao_longo_do_tempo.png"
    )

    figura.savefig(
        caminho_grafico,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figura)

    print("\nGráfico salvo em:")
    print(caminho_grafico)

    print("\nExploração concluída!")


if __name__ == "__main__":
    main()