# Predição de Risco de Defasagem e Abandono Escolar

API e pipeline de Machine Learning para prever risco de abandono escolar a partir de histórico acadêmico PEDE (2022-2024), com monitoramento de drift e artefatos versionados no próprio projeto.

## 1) Visão Geral do Projeto

### Objetivo
Antecipar alunos com maior probabilidade de abandono para apoiar ações preventivas da equipe pedagógica.

### Solução Proposta
A solução é composta por três blocos:

1. Preparação de dados: consolidação de múltiplas abas Excel em uma base refinada (`data/refined/df_base.parquet`).
2. Engenharia de features + target: construção de `ABANDONO` com lógica temporal (presença no ano seguinte), imputação por fase e geração da base de treino (`data/processed/df_trein.parquet`).
3. Serving e monitoramento: API FastAPI para inferência (`/api/predict`), healthcheck (`/health`) e dashboard de drift (`/monitoring`).

### Stack Tecnológica
- Python 3.11
- Pandas / NumPy
- Scikit-learn (RandomForestClassifier)
- FastAPI + Uvicorn
- Evidently (relatório de data drift)
- Pytest
- Docker

## 2) Instruções de Deploy (como subir o ambiente)

## Pré-requisitos
- Python 3.11+
- `pip`
- Docker (opcional, para deploy containerizado)

## Deploy local (ambiente Python)

1. Criar e ativar virtualenv:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependências:
```bash
pip install -r requirements.txt
```

3. (Opcional, recomendado) Regerar dados e treinar modelo:
```bash
python -m src.preprocessing
python -m src.feature_engineering
python -m src.train --min-recall 0.60
```

4. Subir API:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Validar:
```bash
curl http://localhost:8000/health
```

## Deploy com Docker

1. Build da imagem:
```bash
docker build -t abandono-api:latest .
```

2. Run do container:
```bash
docker run --rm -p 8000:8000 abandono-api:latest
```

3. Healthcheck:
```bash
curl http://localhost:8000/health
```

> O `Dockerfile` já está pronto para plataformas como Railway via variável `PORT`.

## 3) Etapas do Pipeline de Machine Learning

## Etapa 1 - Pré-processamento (`src/preprocessing.py`)
- Lê `data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx` (todas as abas).
- Normaliza nomes de colunas para snake_case ASCII.
- Extrai/garante `ANO` por aba, padroniza idade, fase e indicadores.
- Seleciona variáveis base e salva em `data/refined/df_base.parquet`.

Comando:
```bash
python -m src.preprocessing
```

## Etapa 2 - Feature Engineering (`src/feature_engineering.py`)
- Cria alvo `ABANDONO` usando presença em `ANO + 1`.
- Mantém apenas linhas observáveis e fases 1..7 para treino.
- Imputa nulos por mediana dentro de cada fase.
- Cria flags `*_MISSING` para preservar informação de ausência.
- Salva dataset final em `data/processed/df_trein.parquet`.

Comando:
```bash
python -m src.feature_engineering
```

## Etapa 3 - Treino (`src/train.py`)
- Split temporal fixo:
  - Treino: `ANO == 2022`
  - Teste: `ANO == 2023`
- Modelo: `RandomForestClassifier` com pesos de classe.
- Seleção de threshold por recall mínimo (`--min-recall`, padrão `0.60`).
- Salva artefatos em `app/model/`:
  - `random_forest_abandono.joblib`
  - `random_forest_abandono.pkl`
  - `features.pkl`
  - `threshold.pkl`

Comando:
```bash
python -m src.train --min-recall 0.60
```

## Etapa 4 - Avaliação (`src/evaluate.py`)
- Recarrega modelo + features + threshold salvos.
- Avalia no hold-out temporal (`ANO == 2023` por padrão).
- Emite ROC-AUC, PR-AUC, matriz de confusão e classification report.
- Opcional: exporta métricas em JSON.

Comando:
```bash
python -m src.evaluate --metrics-out monitoring/metrics.json
```

## Etapa 5 - Monitoramento de Drift (`monitoring/drift_report.py`)
- Compara dataset de referência vs dataset atual.
- Gera dashboard HTML em `monitoring/drift_report.html`.
- Dashboard disponível na rota da API: `GET /monitoring`.

Comando:
```bash
python -m monitoring.drift_report \
  --reference data/processed/df_trein.parquet \
  --current data/refined/df_base.parquet \
  --output monitoring/drift_report.html
```

## API de Predição

## Endpoint principal
`POST /api/predict`

Payload esperado:
```json
{
  "payload": {
    "IDADE": 13,
    "FASE": 3,
    "DEFASAGEM": 1,
    "IAA": 7.2,
    "IEG": 6.8,
    "IDA": 6.4,
    "IAN": 5.9,
    "IPS": 6.0,
    "IPV": 7.0,
    "NOTA_MAT": 6.3,
    "NOTA_POR": 6.1
  }
}
```

Resposta:
```json
{
  "probability": 0.73,
  "prediction": 1,
  "threshold": 0.58
}
```

## Endpoints auxiliares
- `GET /health`: status da aplicação.
- `GET /monitoring`: dashboard HTML de drift (se já gerado).

## Estrutura do Projeto

```text
app/
  main.py
  routes.py
  logging_config.py
  model/
src/
  preprocessing.py
  feature_engineering.py
  train.py
  evaluate.py
monitoring/
  drift_report.py
  drift_report.html
data/
  raw/
  refined/
  processed/
tests/
```

## Testes

Executar:
```bash
python -m pytest -q
```

Cobertura atual inclui:
- Pré-processamento
- Feature engineering
- Treino/avaliação

## Observações importantes
- O repositório já contém artefatos de modelo em `app/model/`; a API pode subir sem retreino.
- Caso os artefatos estejam ausentes, a rota `/api/predict` retornará erro 500 até que o treino seja executado.
- O arquivo bruto Excel é parte crítica do pipeline e deve existir em `data/raw/`.
