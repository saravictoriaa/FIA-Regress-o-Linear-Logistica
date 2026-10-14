# Fundamentos de Inteligência Artificial — Projetos em Python

Repositório acadêmico desenvolvido para a disciplina de Fundamentos da Inteligência Artificial. Contém dois projetos de aprendizado supervisionado: classificação de sites de phishing com regressão logística e previsão de throughput de rede com regressão linear.

## Projetos

### 1. Detector de phishing — Regressão logística

Utiliza características numéricas de sites para classificá-los como phishing ou legítimos.

**Base de dados:** [Phishing Websites — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)

**Principais resultados:**
- 11.055 registros e 30 características.
- Acurácia no teste: 92,31%.
- AUC-ROC: 0,9782.
- Matriz de confusão, métricas de classificação e curva ROC para análise dos resultados.

Documentação, instruções de execução e limitações: [`projeto_phishing/README.md`](projeto_phishing/README.md).

### 2. Previsão de throughput — Regressão linear

Utiliza medições anteriores de throughput para estimar a próxima medição registrada, em Mbps.

**Base de dados:** [Real-world-bandwidth-traces](https://github.com/confiwent/Real-world-bandwidth-traces).

**Principais resultados no conjunto de teste:**
- 266 medições originais; 263 registros utilizáveis após a preparação.
- 210 registros para treinamento e 53 para teste.
- MAE: 0,1513 Mbps.
- RMSE: 0,2434 Mbps.
- R²: 0,7599.

A comparação com a referência simples — repetir a última medição — mostrou MAE de 0,1376 Mbps e RMSE de 0,2426 Mbps. Portanto, a regressão linear não superou essa referência nas duas métricas de erro.

Documentação, instruções de execução e limitações: [`projeto_banda/README.md`](projeto_banda/README.md).

## Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- Matplotlib
- Joblib
- ucimlrepo

## Estrutura do repositório

```text
fundamentos-ia/
├── projeto_phishing/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── src/
│   └── README.md
├── projeto_banda/
│   ├── data/
│   ├── models/
│   ├── results/
│   ├── src/
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

Os diretórios de dados e modelos são gerados ou preenchidos durante a execução. Arquivos ignorados pelo Git, como o ambiente virtual e os modelos treinados, não são incluídos no repositório.

## Como configurar o ambiente

Os comandos abaixo consideram Windows e PowerShell, executados na raiz do repositório.

### 1. Criar e ativar o ambiente virtual

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 3. Executar os projetos

Para explorar, preparar e treinar o detector de phishing:

```powershell
python projeto_phishing\src\explorar_dados.py
python projeto_phishing\src\preparar_dados.py
python projeto_phishing\src\treinar_modelo.py
```

Para fazer uma previsão com o modelo de phishing já treinado:

```powershell
python projeto_phishing\src\prever_phishing.py
```

Para preparar, comparar variáveis e treinar o modelo de throughput:

```powershell
python projeto_banda\src\explorar_dados.py
python projeto_banda\src\preparar_dados.py
python projeto_banda\src\comparar_variaveis.py
python projeto_banda\src\treinar_modelo.py
```

Para fazer uma previsão de throughput com o modelo treinado:

```powershell
python projeto_banda\src\prever_banda.py
```

**Observações:** os scripts de exploração e preparação do phishing precisam de conexão com a internet para carregar a base da UCI. No projeto de banda, o arquivo original precisa estar em `projeto_banda/data/raw/norway_bus_1.txt`; o README desse projeto contém as instruções para baixá-lo.

## Objetivo acadêmico

Os projetos demonstram etapas essenciais de um fluxo de Machine Learning: exploração e preparação dos dados, treinamento, avaliação com métricas apropriadas, geração de gráficos e utilização de modelos treinados para previsões.

Os resultados são específicos das bases e divisões utilizadas. Eles não garantem o mesmo desempenho em outros dados ou em aplicações reais.