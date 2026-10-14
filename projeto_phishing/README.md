# Detector de Phishing com Regressão Logística

## 1. Objetivo

Este projeto foi desenvolvido para a disciplina de Fundamentos da Inteligência Artificial. O objetivo é utilizar regressão logística para classificar registros de sites como **phishing** ou **legítimos**, aplicando conceitos de aprendizado supervisionado com Python.

O projeto inclui exploração e preparação dos dados, treinamento do modelo, avaliação dos resultados, geração de gráficos e execução de previsões com o modelo treinado.

## 2. Base de dados

Foi utilizada a base [Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites), disponibilizada no UCI Machine Learning Repository.

A base carregada no projeto contém:

- 11.055 registros;
- 30 características de entrada;
- uma variável-alvo chamada `result`;
- nenhum valor ausente identificado na exploração inicial.

A variável `result` utiliza a seguinte codificação:

| Valor | Classe |
|---|---|
| `-1` | Phishing |
| `1` | Legítimo |

As características já estão extraídas e representadas numericamente. Por isso, o modelo não analisa diretamente uma URL digitada pelo usuário, mas utiliza os atributos disponíveis na base.

## 3. Tecnologias utilizadas

- **Python:** desenvolvimento dos scripts.
- **Pandas e NumPy:** manipulação dos dados e operações numéricas.
- **Scikit-learn:** preparação, treinamento e avaliação do modelo.
- **Matplotlib:** geração dos gráficos.
- **Joblib:** salvamento e carregamento do modelo.
- **ucimlrepo:** acesso à base da UCI.

## 4. Como o projeto funciona

O desenvolvimento foi dividido em quatro etapas.

### 4.1. Exploração dos dados

Arquivo: `src/explorar_dados.py`

Carrega a base e apresenta suas dimensões, características, valores ausentes, distribuição das classes e informações sobre registros duplicados.

Na análise, foram encontradas 5.206 ocorrências repetidas além da primeira ocorrência de cada combinação de características e classe. Também foram identificados 64 perfis de características associados às duas classes.

Isso mostra que existem particularidades nos dados que precisam ser consideradas na avaliação do modelo.

### 4.2. Preparação dos dados

Arquivo: `src/preparar_dados.py`

Os dados foram separados em conjuntos de treinamento e teste utilizando `StratifiedGroupKFold`, com cinco divisões. O código agrupa registros com as mesmas características para evitar que um perfil idêntico apareça nos dois conjuntos.

A divisão resultou em:

| Conjunto | Registros | Proporção |
|---|---:|---:|
| Treinamento | 8.843 | 80% |
| Teste | 2.212 | 20% |

A distribuição das classes foi preservada em aproximadamente 44,3% para phishing e 55,7% para sites legítimos nos dois conjuntos.

Os arquivos gerados são `data/processed/treino.csv` e `data/processed/teste.csv`.

### 4.3. Treinamento e avaliação

Arquivo: `src/treinar_modelo.py`

O modelo utiliza uma `Pipeline` composta por:

1. `StandardScaler`, que padroniza as características;
2. `LogisticRegression`, que aprende a relacionar as características às classes.

O treinamento utiliza apenas o conjunto de treinamento. Depois, o desempenho é avaliado com os dados reservados para teste.

O script apresenta métricas de classificação, gera três gráficos e salva o modelo treinado em `models/regressao_logistica.joblib`.

### 4.4. Previsão

Arquivo: `src/prever_phishing.py`

Carrega o modelo salvo e permite selecionar um registro do conjunto de teste. O programa apresenta a classe prevista, as probabilidades estimadas e a classe real registrada na base.

Essa etapa demonstra como reutilizar um modelo já treinado sem precisar executar novamente o treinamento.

## 5. Resultados

Na avaliação realizada com os 2.212 registros de teste, foram obtidos os seguintes resultados:

| Métrica | Phishing | Legítimo |
|---|---:|---:|
| Precision | 0,91 | 0,93 |
| Recall | 0,91 | 0,93 |
| F1-score | 0,91 | 0,93 |

**Acurácia geral:** 92,31%

**AUC-ROC:** 0,9782

Os valores de precision, recall e F1-score estão arredondados para duas casas decimais.

### 5.1. Matriz de confusão

| Classe real | Previsto como phishing | Previsto como legítimo |
|---|---:|---:|
| Phishing | 896 | 84 |
| Legítimo | 86 | 1.146 |

Dos 2.212 registros de teste, 2.042 foram classificados corretamente e 170 incorretamente.

Os 84 falsos negativos são exemplos de phishing classificados como legítimos. Já os 86 falsos positivos são sites legítimos classificados como phishing.

Esses resultados mostram por que é importante analisar os tipos de erro, e não apenas a acurácia.

### 5.2. Gráficos gerados

Os gráficos ficam salvos em `results/graficos/`:

- **`matriz_confusao.png`:** apresenta os acertos e erros de cada classe.
- **`metricas_classificacao.png`:** compara precision, recall e F1-score entre as classes.
- **`curva_roc.png`:** mostra a capacidade de discriminação do modelo para diferentes limiares de classificação.

O AUC-ROC de 0,9782 indica boa capacidade de distinguir as classes na avaliação realizada. Esse valor não representa uma porcentagem de acertos.

## 6. Estrutura do projeto

```text
projeto_phishing/
├── data/
│   └── processed/
│       ├── treino.csv
│       └── teste.csv
├── models/
│   └── regressao_logistica.joblib
├── results/
│   └── graficos/
│       ├── matriz_confusao.png
│       ├── metricas_classificacao.png
│       └── curva_roc.png
├── src/
│   ├── explorar_dados.py
│   ├── preparar_dados.py
│   ├── treinar_modelo.py
│   └── prever_phishing.py
└── README.md
```

Os arquivos dentro de `data/processed/` e `models/` são gerados durante a execução. Conforme as regras do `.gitignore`, eles não precisam ser incluídos no versionamento. Os gráficos são gerados pelo script de treinamento.

## 7. Como executar

Os comandos abaixo consideram o uso do PowerShell no Windows, a partir da raiz do repositório `fundamentos-ia`.

### 7.1. Ativar o ambiente virtual

```powershell
.\.venv\Scripts\Activate.ps1
```

### 7.2. Instalar as dependências

```powershell
python -m pip install -r requirements.txt
```

### 7.3. Explorar a base

```powershell
python projeto_phishing\src\explorar_dados.py
```

### 7.4. Preparar os dados

```powershell
python projeto_phishing\src\preparar_dados.py
```

### 7.5. Treinar o modelo e gerar os gráficos

```powershell
python projeto_phishing\src\treinar_modelo.py
```

### 7.6. Fazer uma previsão

```powershell
python projeto_phishing\src\prever_phishing.py
```

Os scripts que carregam a base pela UCI precisam de conexão com a internet. Os scripts de treinamento e previsão dependem dos arquivos gerados nas etapas anteriores.

## 8. Solução de problemas: OpenBLAS

Durante o desenvolvimento, ocorreu o erro:

```text
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

O problema foi contornado limitando o paralelismo das bibliotecas numéricas no PowerShell.

Com o ambiente virtual ativado, execute os comandos abaixo no mesmo terminal em que rodará o programa:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OPENBLAS_DEFAULT_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
```

Em seguida, execute novamente o treinamento:

```powershell
python projeto_phishing\src\treinar_modelo.py
```

Essas configurações são temporárias e valem para a sessão atual do PowerShell. Se o erro persistir, verifique se as bibliotecas podem ser importadas:

```powershell
python -c "import numpy; import sklearn; print('NumPy:', numpy.__version__); print('Scikit-learn:', sklearn.__version__)"
```

A configuração resolveu o problema observado neste ambiente, mas outras causas de erro do OpenBLAS podem exigir uma investigação específica.

## 9. Limitações

O projeto possui algumas limitações:

- O modelo foi treinado e avaliado com uma única base de dados.
- Foram encontrados perfis de características associados às duas classes, o que limita a distinção de alguns exemplos.
- O desempenho obtido não garante os mesmos resultados em outros conjuntos de dados ou em sites reais.
- O modelo utiliza características previamente extraídas e não classifica diretamente URLs fornecidas pelo usuário.
- As probabilidades estimadas não garantem que um site seja seguro ou malicioso.

Uma aplicação real exigiria validação adicional com dados externos e uma etapa confiável de extração das características utilizadas pelo modelo.

## 10. Conclusão

O projeto aplicou regressão logística a um problema de classificação binária, passando pelas etapas de exploração e preparação dos dados, treinamento, avaliação e previsão.

Na divisão de teste utilizada, o modelo obteve 92,31% de acurácia e AUC-ROC de 0,9782. Os gráficos ajudaram a analisar o desempenho e identificar os tipos de erro cometidos.

O trabalho permitiu praticar conceitos fundamentais de aprendizado supervisionado e desenvolver um fluxo reproduzível de Machine Learning com Python, considerando as limitações da base utilizada.