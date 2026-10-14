# Previsão de Throughput de Rede com Regressão Linear

## 1. Objetivo

Projeto desenvolvido para a disciplina de Fundamentos da Inteligência Artificial, com o objetivo de aplicar regressão linear para estimar a próxima medição de throughput de uma conexão de rede a partir de uma medição anterior.

O trabalho utiliza Python para explorar os dados, preparar as variáveis, treinar o modelo, avaliar as previsões e gerar gráficos.

## 2. Base de dados

Foi utilizado o arquivo `norway_bus_1`, disponível no repositório público [Real-world-bandwidth-traces](https://github.com/confiwent/Real-world-bandwidth-traces).

O repositório reúne séries de medições reais de throughput de rede. Os arquivos processados utilizam duas colunas: tempo, em segundos, e vazão, em Mbit/s.

**Arquivo utilizado:** `cooked_3gp/norway_bus_1`

Na exploração realizada neste projeto, o arquivo apresentou:

- 266 medições;
- duas colunas numéricas: `tempo_segundos` e `throughput_mbps`;
- nenhum valor ausente;
- nenhuma linha duplicada;
- tempos em ordem crescente.

A vazão média observada foi de aproximadamente 3,34 Mbps, com valores entre 0,37 Mbps e 4,79 Mbps.

A série apresenta oscilações e os intervalos entre medições não são constantes.

## 3. Tecnologias utilizadas

- **Python:** desenvolvimento dos scripts.
- **Pandas:** leitura e preparação dos dados.
- **NumPy:** operações numéricas e cálculo de métricas.
- **Scikit-learn:** treinamento da regressão linear e avaliação das previsões.
- **Matplotlib:** geração dos gráficos.
- **Joblib:** salvamento e carregamento do modelo treinado.

## 4. Como o projeto funciona

O desenvolvimento foi dividido em cinco etapas.

### 4.1. Exploração dos dados

Arquivo: `src/explorar_dados.py`

Carrega o arquivo original e apresenta as dimensões, as primeiras e últimas linhas, os tipos das colunas, os valores ausentes, as estatísticas da vazão e os intervalos entre medições.

Também gera o gráfico `results/graficos/vazao_ao_longo_do_tempo.png`, utilizado para observar as oscilações do throughput.

### 4.2. Preparação dos dados

Arquivo: `src/preparar_dados.py`

Foram criadas três variáveis com medições anteriores:

- `throughput_t_1`: última medição conhecida;
- `throughput_t_2`: medição anterior à última;
- `throughput_t_3`: terceira medição anterior.

As três primeiras observações foram descartadas porque não possuíam histórico suficiente para preencher todas as variáveis.

Dos 266 registros originais, 263 foram utilizados na preparação. A divisão cronológica resultou em 210 registros para treinamento e 53 para teste, aproximadamente 80% e 20%, respectivamente.

A ordem temporal foi mantida para evitar que observações futuras fossem utilizadas no treinamento de um modelo avaliado em observações passadas.

### 4.3. Comparação de variáveis

Arquivo: `src/comparar_variaveis.py`

Foram comparadas três configurações de regressão linear: uma, duas e três medições anteriores como variáveis de entrada.

A comparação foi realizada utilizando uma divisão cronológica interna do conjunto de treinamento, com 168 registros para ajuste e 42 para validação.

| Configuração | MAE (Mbps) | RMSE (Mbps) |
|---|---:|---:|
| Uma medição anterior | 0,1818 | 0,2761 |
| Duas medições anteriores | 0,1929 | 0,2847 |
| Três medições anteriores | 0,1942 | 0,2889 |
| Referência: repetir a última medição | 0,1690 | 0,2829 |

Entre as configurações de regressão linear, a que utiliza uma medição anterior apresentou o menor MAE e o menor RMSE. A referência simples, entretanto, teve o menor MAE entre todas as opções.

O gráfico `results/graficos/comparacao_variaveis.png` apresenta essa comparação.

### 4.4. Treinamento e avaliação

Arquivo: `src/treinar_modelo.py`

O modelo final utiliza somente `throughput_t_1`, ou seja, a última medição conhecida.

A regressão linear é ajustada com os dados de treinamento e avaliada nos 53 registros reservados para teste. O script calcula MAE, RMSE e R², compara o resultado com uma referência simples e salva o modelo treinado.

Resultados obtidos no conjunto de teste:

| Métrica | Regressão linear | Referência simples |
|---|---:|---:|
| MAE | 0,1513 Mbps | 0,1376 Mbps |
| RMSE | 0,2434 Mbps | 0,2426 Mbps |
| R² | 0,7599 | — |

A referência simples prevê que a próxima medição será igual à última medição conhecida.

Os resultados mostram que a regressão linear conseguiu modelar parte das variações da série, mas não superou a referência simples nas métricas MAE e RMSE neste teste.

O R² de 0,7599 indica que o modelo explica parte importante da variação em comparação com uma previsão constante baseada na média do alvo. Essa métrica utiliza uma referência diferente daquela empregada na comparação com a última medição.

O gráfico `results/graficos/comparacao_real_previsto.png` compara a vazão observada com as previsões produzidas pelo modelo e pela referência simples.

O gráfico `results/graficos/comparacao_metricas.png`, quando gerado pelo script, compara visualmente os erros dos dois métodos.

### 4.5. Previsão com o modelo salvo

Arquivo: `src/prever_banda.py`

Carrega o modelo salvo em `models/regressao_linear.joblib` e solicita ao usuário a última medição de throughput em Mbps.

Em um teste realizado com a entrada de 3,5 Mbps, o modelo estimou 3,4758 Mbps para a próxima medição.

Essa funcionalidade demonstra como utilizar um modelo já treinado para obter uma previsão sem executar novamente o treinamento.

## 5. Estrutura do projeto

```text
projeto_banda/
├── data/
│   ├── raw/
│   │   └── norway_bus_1.txt
│   └── processed/
│       ├── treino.csv
│       └── teste.csv
├── models/
│   └── regressao_linear.joblib
├── results/
│   └── graficos/
│       ├── vazao_ao_longo_do_tempo.png
│       ├── comparacao_variaveis.png
│       ├── comparacao_real_previsto.png
│       └── comparacao_metricas.png
├── src/
│   ├── explorar_dados.py
│   ├── preparar_dados.py
│   ├── comparar_variaveis.py
│   ├── treinar_modelo.py
│   └── prever_banda.py
└── README.md
```

Os arquivos em `data/processed/` e `models/` são gerados durante a execução e estão excluídos do versionamento pelo `.gitignore`. O arquivo de dados original também não precisa ser enviado ao GitHub.

## 6. Como executar

Os comandos abaixo consideram o uso do PowerShell no Windows, a partir da raiz do repositório `fundamentos-ia`.

### 6.1. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### 6.2. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 6.3. Baixar a base de dados

O arquivo original está disponível publicamente no repositório indicado acima. Execute o comando abaixo a partir da raiz do projeto para baixá-lo para a pasta esperada pelos scripts:

```powershell
New-Item -ItemType Directory -Force -Path "projeto_banda\data\raw"

Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/confiwent/Real-world-bandwidth-traces/master/cooked_3gp/norway_bus_1" `
  -OutFile "projeto_banda\data\raw\norway_bus_1.txt"
```

O download exige conexão com a internet.

### 6.4. Explorar os dados

```powershell
python projeto_banda\src\explorar_dados.py
```

### 6.5. Preparar os dados

```powershell
python projeto_banda\src\preparar_dados.py
```

### 6.6. Comparar as variáveis de entrada

```powershell
python projeto_banda\src\comparar_variaveis.py
```

### 6.7. Treinar e avaliar o modelo

```powershell
python projeto_banda\src\treinar_modelo.py
```

O script treina o modelo, apresenta as métricas e gera os gráficos de avaliação.

### 6.8. Fazer uma previsão

```powershell
python projeto_banda\src\prever_banda.py
```

O programa solicitará a última medição conhecida e exibirá a previsão de throughput para a próxima observação.

## 7. Limitações

O projeto possui algumas limitações:

- Foi utilizada somente uma série de medições, com 266 registros.
- Os intervalos de tempo entre observações são irregulares.
- A previsão corresponde à próxima medição da série, e não a um horizonte temporal fixo.
- O modelo utiliza apenas a última medição como variável de entrada.
- A regressão linear não superou a referência simples nas métricas MAE e RMSE do conjunto de teste.
- Os resultados não garantem o mesmo desempenho em outras redes ou condições de conexão.

O objetivo principal é demonstrar, de forma didática, o fluxo de preparação, treinamento e avaliação de um modelo de regressão linear.

## 8. Conclusão

O projeto aplicou regressão linear para estimar o throughput da próxima medição utilizando dados históricos de uma série pública.

Foram realizadas a exploração e a preparação dos dados, a comparação entre diferentes variáveis de entrada, o treinamento, a avaliação e a utilização do modelo salvo para realizar previsões.

No conjunto de teste, a regressão linear obteve MAE de 0,1513 Mbps, RMSE de 0,2434 Mbps e R² de 0,7599. Apesar do R² positivo, a estratégia de repetir a última medição apresentou erros menores em MAE e RMSE.

O resultado demonstra que a escolha de um algoritmo deve ser acompanhada de avaliação e comparação com referências simples. O desempenho observado é válido para a série e a divisão utilizadas, não constituindo garantia de previsão para qualquer rede.