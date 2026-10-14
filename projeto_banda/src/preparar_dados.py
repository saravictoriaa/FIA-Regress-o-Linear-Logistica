
from pathlib import Path

import pandas as pd


def main():
    # Localiza os arquivos do projeto.
    pasta_projeto = Path(__file__).resolve().parents[1]

    caminho_entrada = (
        pasta_projeto / "data" / "raw" / "norway_bus_1.txt"
    )

    pasta_saida = pasta_projeto / "data" / "processed"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print("Carregando as medições de banda...")

    # Lê o arquivo original.
    dados = pd.read_csv(
        caminho_entrada,
        sep=r"\s+",
        header=None,
        names=["tempo_segundos", "throughput_mbps"],
    )

    # Mantém o horário da medição-alvo para análises futuras.
    # Obs.:Esse horário não será usado como entrada do modelo.
    supervisionado = pd.DataFrame({
        "tempo_alvo_segundos": dados["tempo_segundos"],
        "throughput_mbps": dados["throughput_mbps"],
    })

    # Cria três entradas com as medições anteriores.
    for atraso in range(1, 4):
        coluna = f"throughput_t_{atraso}"
        supervisionado[coluna] = dados["throughput_mbps"].shift(atraso)

    # As três primeiras linhas não possuem histórico suficiente.
    supervisionado = supervisionado.dropna().reset_index(drop=True)

    if len(supervisionado) < 10:
        raise ValueError(
            "Há poucos registros para preparar os conjuntos de dados."
        )

    # Divide cronologicamente: primeiros 80% para treino,
    # últimos 20% para teste, não embaralhamos a série temporal.
    ponto_corte = int(len(supervisionado) * 0.8)

    treino = supervisionado.iloc[:ponto_corte].copy()
    teste = supervisionado.iloc[ponto_corte:].copy()

    # Salva os conjuntos preparados.
    treino.to_csv(pasta_saida / "treino.csv", index=False)
    teste.to_csv(pasta_saida / "teste.csv", index=False)

    print("\n--- Resultado da preparação ---")
    print(f"Medições originais: {len(dados)}")
    print(f"Registros utilizáveis: {len(supervisionado)}")
    print(f"Registros para treino: {len(treino)}")
    print(f"Registros para teste: {len(teste)}")

    print("\nVariáveis de entrada do modelo:")
    print("throughput_t_1")
    print("throughput_t_2")
    print("throughput_t_3")

    print("\nVariável que será prevista:")
    print("throughput_mbps")

    print("\nPrimeiras linhas dos dados preparados:")
    print(supervisionado.head())

    print("\nArquivos salvos em:")
    print(pasta_saida)

    print("\nPreparação concluída!")


if __name__ == "__main__":
    main()
