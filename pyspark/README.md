# Churn Prediction Pipeline — PySpark ML

Système complet de **prédiction de churn client** (départ des clients) basé sur **PySpark ML**.  
Le pipeline couvre toutes les étapes : téléchargement des données → entraînement → scoring → monitoring → API REST.

---

## À quoi ça sert ?

> **Le churn** = un client qui arrête d'acheter. Ce projet prédit *quels clients risquent de partir* à partir de leur historique d'achats.

Le pipeline :
1. **Télécharge** le dataset de transactions e-commerce (UCI Online Retail II — ~1 million de lignes)
2. **Nettoie** les données (retours, prix nuls, doublons, clients anonymes)
3. **Calcule les features RFM** par client (Recency, Frequency, Monetary + 4 autres)
4. **Entraîne un modèle** (Random Forest, GBT ou Régression Logistique) pour prédire le churn
5. **Score tous les clients** avec une probabilité de churn [0–1]
6. **Détecte le data drift** (PSI) pour surveiller si les données ont changé
7. **Expose une API REST** (FastAPI) pour scorer de nouveaux clients en temps réel

---

## Architecture

```
Données brutes (CSV)
       │
       ▼
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  src/data/      │────▶│  src/feature/    │────▶│  src/models/      │
│  download.py    │     │  builder.py      │     │  train.py         │
│  cleaning.py    │     │  (features RFM)  │     │  evaluate.py      │
│  schema.py      │     └──────────────────┘     │  predict.py       │
└─────────────────┘                              │  pipeline.py      │
                                                 └───────────────────┘
                                                          │
                              ┌───────────────────────────┼───────────────────────────┐
                              ▼                           ▼                           ▼
                   ┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
                   │  src/           │        │  src/monitoring/ │        │  api/            │
                   │  orchestration/ │        │  drift.py (PSI)  │        │  FastAPI REST    │
                   │  pipeline.py   │        │  report.py       │        │  /predict        │
                   └─────────────────┘        └──────────────────┘        └──────────────────┘
```

---

## Installation

### Prérequis

| Outil | Version minimale |
|-------|-----------------|
| Python | 3.10+ |
| Java (JDK) | 11 ou 17 (requis par Spark) |
| PySpark | 3.5+ |

### 1. Cloner et créer l'environnement

```bash
# Créer un environnement virtuel
python -m venv .env
source .env/bin/activate        # Linux / Mac
.env\Scripts\activate           # Windows PowerShell

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Vérifier Java

```bash
java -version   # doit afficher Java 11 ou 17
```

> Sur Windows : télécharger [Eclipse Temurin JDK 17](https://adoptium.net/) et définir `JAVA_HOME`.

---

## Démarrage rapide

### Option A — Pipeline complet en 1 commande

```bash
python -m cli.pipeline
```

Cette commande enchaîne automatiquement les 4 étapes :
téléchargement → entraînement → scoring → monitoring.

### Option B — Étape par étape

```bash
python -m cli.data_download   # étape 1 : données
python -m cli.train           # étape 2 : entraînement
python -m cli.predict         # étape 3 : prédictions
python -m cli.serve           # étape 4 : API REST
```

---

## Structure du projet

```
pyspark/
│
├── cli/                        ← Scripts de lancement (point d'entrée)
│   ├── pipeline.py             ← Pipeline complet (tout en 1)
│   ├── data_download.py        ← Téléchargement + nettoyage + features
│   ├── train.py                ← Entraînement du modèle
│   ├── predict.py              ← Prédictions batch
│   └── serve.py                ← Lancement de l'API FastAPI
│
├── src/                        ← Modules métier (réutilisables)
│   ├── config.py               ← Configuration centrale (dataclasses)
│   ├── spark/
│   │   └── session.py          ← Factory SparkSession
│   ├── data/
│   │   ├── download.py         ← Téléchargement UCI + chargement CSV
│   │   ├── cleaning.py         ← Nettoyage des transactions
│   │   ├── schema.py           ← Schémas Spark (TRANSACTIONS_SCHEMA…)
│   │   └── eda.py              ← Analyse exploratoire (EDA)
│   ├── feature/
│   │   └── builder.py          ← Calcul features RFM + label churn
│   ├── models/
│   │   ├── pipeline.py         ← Construction du Pipeline MLlib
│   │   ├── train.py            ← Entraînement + cross-validation
│   │   ├── evaluate.py         ← Métriques AUC, F1, confusion matrix
│   │   └── predict.py          ← Scoring + top-N à risque
│   ├── orchestration/
│   │   ├── steps.py            ← Étapes atomiques (StepResult)
│   │   └── pipeline.py         ← run_full_pipeline() orchestrateur
│   └── monitoring/
│       ├── drift.py            ← Calcul PSI (data drift)
│       └── report.py           ← Génération rapport monitoring
│
├── api/                        ← API FastAPI
│   ├── main.py                 ← Application + lifespan + middleware
│   ├── dependencies.py         ← AppState (Spark singleton)
│   ├── schemas/
│   │   ├── request.py          ← CustomerFeatures, BatchPredictRequest
│   │   └── response.py         ← ChurnPrediction, BatchPredictResponse…
│   └── routers/
│       ├── health.py           ← GET /health  GET /status
│       └── predict.py          ← POST /predict  POST /predict/batch  GET /predict/top-risk
│
├── tests/                      ← Suite de tests (pytest)
│   ├── conftest.py             ← Fixtures partagées (Spark, data, modèle)
│   ├── unit/                   ← Tests unitaires (~40 tests)
│   │   ├── test_schema.py
│   │   ├── test_cleaning.py
│   │   ├── test_builder.py
│   │   ├── test_pipeline_build.py
│   │   ├── test_evaluate.py
│   │   └── test_predict.py
│   └── integration/            ← Tests d'intégration (~20 tests)
│       ├── test_data_pipeline.py
│       └── test_train_predict.py
│
├── data/                       ← Données (générées automatiquement)
│   ├── raw/                    ← CSV brut UCI téléchargé
│   └── processed/features/     ← Features Parquet (après data_download)
│
├── output/                     ← Artefacts ML (générés automatiquement)
│   ├── model/                  ← PipelineModel PySpark sauvegardé
│   ├── metrics.json            ← AUC, F1, accuracy, confusion matrix
│   ├── predictions.csv         ← Tous les clients scorés
│   ├── top_at_risk.csv         ← Top-20 clients les plus à risque
│   ├── pipeline_run.json       ← Résumé JSON de toutes les étapes
│   └── monitoring/
│       └── drift_report.json   ← Rapport PSI de data drift
│
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Scripts CLI — Détail

### `cli/pipeline.py` — Pipeline complet (recommandé)

Lance les 4 étapes en séquence avec un seul script.

```bash
# Lancement standard
python -m cli.pipeline

# Options fréquentes
python -m cli.pipeline --skip-download            # CSV déjà téléchargé
python -m cli.pipeline --skip-train               # Modèle déjà entraîné → scoring + monitoring uniquement
python -m cli.pipeline --force-download           # Re-télécharger + tout recalculer
python -m cli.pipeline --env production           # Utiliser la config production
python -m cli.pipeline --algo gbt --cv            # GBT avec cross-validation
python -m cli.pipeline --no-monitoring            # Sans l'étape de monitoring PSI
python -m cli.pipeline --cutoff 2011-09-01        # Changer la date de cutoff RFM
python -m cli.pipeline --psi-warning 0.05         # Seuil PSI WARNING plus strict
```

**Ce que ça produit :**

```
[1/4] DataPipeline   → data/processed/features/ (Parquet)
[2/4] Training       → output/model/  +  output/metrics.json
[3/4] Scoring        → output/predictions.csv  +  output/top_at_risk.csv
[4/4] Monitoring     → output/monitoring/drift_report.json
```

---

### `cli/data_download.py` — Données seulement

Télécharge le dataset, nettoie les transactions et calcule les features RFM.

```bash
python -m cli.data_download
python -m cli.data_download --cutoff 2011-09-01 --horizon-days 90
python -m cli.data_download --skip-download    # CSV déjà présent, recalcule uniquement les features
python -m cli.data_download --force            # Re-télécharge même si CSV existe
```

**Étapes internes :**
| Étape | Action | Source |
|-------|--------|--------|
| 1/4 | Téléchargement ZIP UCI (~45 MB) | `src/data/download.py` |
| 2/4 | Chargement CSV → Spark DataFrame | `src/data/download.py` |
| 3/4 | Nettoyage (retours, prix nuls, doublons) + rapport qualité | `src/data/cleaning.py` |
| 4/4 | Features RFM + label churn → Parquet | `src/feature/builder.py` |

**Features calculées par client :**

| Feature | Description |
|---------|-------------|
| `recency` | Jours depuis le dernier achat |
| `frequency` | Nombre de factures distinctes |
| `monetary` | Revenu total cumulé (€) |
| `avg_basket` | Valeur moyenne du panier |
| `n_products` | Nombre de références produits distincts |
| `n_countries` | Nombre de pays de livraison distincts |
| `tenure_days` | Ancienneté client en jours |
| `label` | **1** = churné (inactif 90j après cutoff), **0** = retenu |

---

### `cli/train.py` — Entraînement seulement

```bash
python -m cli.train
python -m cli.train --algo gbt --num-trees 100 --max-depth 10
python -m cli.train --cv                          # Avec cross-validation 3 plis
python -m cli.train --features data/processed/features  # Chemin Parquet custom
```

**Options algorithmes :**

| `--algo` | Algorithme | Paramètres |
|----------|------------|------------|
| `random_forest` | Random Forest (défaut) | `--num-trees`, `--max-depth` |
| `gbt` | Gradient Boosted Trees | `--num-trees`, `--max-depth` |
| `logistic` | Régression Logistique | — |

**Sorties :**
- `output/model/` — PipelineModel PySpark sauvegardé
- `output/metrics.json` — AUC, F1, accuracy, precision, recall, confusion matrix, feature importances

---

### `cli/predict.py` — Prédictions seulement

```bash
python -m cli.predict
python -m cli.predict --model-dir output/model --top-n 50
python -m cli.predict --threshold 0.6            # Seuil churn plus strict
python -m cli.predict --output resultats.csv
```

**Sortie console :**
```
  #     CustomerID      Churn Proba    Prediction
  ───── ────────────── ────────────── ────────────
  1     C17850              0.9423      🔴 CHURN
  2     C12583              0.8891      🔴 CHURN
  3     C14096              0.7234      🔴 CHURN
  ...
```

---

### `cli/serve.py` — API FastAPI

```bash
python -m cli.serve
python -m cli.serve --port 8080 --model-dir output/model
python -m cli.serve --reload                      # Mode développement (hot-reload)
python -m cli.serve --workers 4 --env production  # Mode production
```

---

## API FastAPI

### Démarrer le serveur

```bash
python -m cli.serve
# → http://localhost:8000
# → Documentation : http://localhost:8000/docs
```

### Endpoints disponibles

| Méthode | Route | Description |
|---------|-------|-------------|
| `GET` | `/` | Informations sur l'API |
| `GET` | `/health` | État du serveur (Spark actif ? modèle chargé ?) |
| `GET` | `/status` | Configuration complète + uptime |
| `POST` | `/predict` | Score **1 client** |
| `POST` | `/predict/batch` | Score **1 à 1 000 clients** |
| `GET` | `/predict/top-risk?n=20` | Top-N clients les plus à risque |

### Exemples de requêtes

**Score 1 client :**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "C12345",
    "recency": 90,
    "frequency": 2,
    "monetary": 80.0,
    "avg_basket": 40.0,
    "n_products": 3,
    "n_countries": 1,
    "tenure_days": 200
  }'
```

**Réponse :**
```json
{
  "prediction": {
    "customer_id": "C12345",
    "churn_proba": 0.823,
    "prediction": 1,
    "risk_level": "HIGH",
    "scored_at": "2026-05-21T10:00:00Z"
  },
  "processing_time_ms": 142.5
}
```

**Niveaux de risque :**

| `risk_level` | `churn_proba` | Action recommandée |
|---|---|---|
| 🟢 `LOW` | < 0.33 | Aucune action |
| 🟡 `MEDIUM` | 0.33 – 0.66 | Surveiller |
| 🔴 `HIGH` | > 0.66 | Campagne de rétention urgente |

**Score batch (10 clients) :**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {"customer_id": "C001", "recency": 90, "frequency": 2, "monetary": 80.0,
       "avg_basket": 40.0, "n_products": 3, "n_countries": 1, "tenure_days": 200},
      {"customer_id": "C002", "recency": 5, "frequency": 15, "monetary": 900.0,
       "avg_basket": 60.0, "n_products": 20, "n_countries": 2, "tenure_days": 500}
    ],
    "threshold": 0.5
  }'
```

**Top-20 clients à risque (depuis le Parquet des features) :**
```bash
curl "http://localhost:8000/predict/top-risk?n=20"
```

---

## Monitoring — Data Drift (PSI)

Le module de monitoring calcule le **PSI (Population Stability Index)** pour détecter si les données ont changé depuis l'entraînement.

### Interprétation du PSI

| PSI | Statut | Signification | Action |
|-----|--------|---------------|--------|
| `< 0.10` | 🟢 **STABLE** | Distribution stable | Aucune action |
| `0.10 – 0.25` | 🟡 **WARNING** | Changement modéré | Surveiller, planifier ré-entraînement |
| `≥ 0.25` | 🔴 **CRITICAL** | Drift majeur | Ré-entraîner immédiatement |

### Rapport de monitoring (console)

```
══════════════════════════════════════════════════════════════════════
 RAPPORT DE MONITORING — DATA DRIFT (PSI)
══════════════════════════════════════════════════════════════════════
  Généré le     : 2026-05-21T10:00:00+00:00
  Statut global : 🟢  STABLE
  Seuils        : WARNING ≥ 0.10   CRITICAL ≥ 0.25

  Feature          PSI      Statut      Distribution drift
  ──────────────── ───────  ──────────  ────────────────────
  recency          0.0312   🟢 STABLE   ██░░░░░░░░░░░░░░░░░░
  frequency        0.0089   🟢 STABLE   █░░░░░░░░░░░░░░░░░░░
  monetary         0.1450   🟡 WARNING  ████████░░░░░░░░░░░░
  avg_basket       0.0210   🟢 STABLE   ░░░░░░░░░░░░░░░░░░░░
  ...
  churn_proba      0.0421   🟢 STABLE   ████░░░░░░░░░░░░░░░░  ← score drift
```

---

## Tests

### Lancer les tests

```bash
# Tous les tests
pytest

# Tests unitaires seulement (rapides, ~30 secondes)
pytest -m unit

# Tests d'intégration (plus lents, ~2-5 minutes avec Spark)
pytest -m integration

# Un fichier précis
pytest tests/unit/test_cleaning.py -v

# Avec couverture de code
pytest --cov=src --cov-report=term-missing
```

### Organisation des tests (~60 tests)

| Fichier | Ce qui est testé | Nb tests |
|---------|-----------------|----------|
| `test_schema.py` | Schémas Spark (types, nullabilité) | 11 |
| `test_cleaning.py` | Nettoyage transactions | 12 |
| `test_builder.py` | Features RFM + labels churn | 12 |
| `test_pipeline_build.py` | Construction Pipeline MLlib | 13 |
| `test_evaluate.py` | Métriques AUC, F1, confusion matrix | 11 |
| `test_predict.py` | Scoring + top-N clients | 10 |
| `test_data_pipeline.py` *(intégration)* | CSV→clean→features→Parquet | 10 |
| `test_train_predict.py` *(intégration)* | Train→save→load→predict | 10 |

> Les tests d'intégration démarrent une vraie SparkSession locale. La fixture `spark` est **session-scoped** (créée une seule fois) pour éviter des démarrages répétitifs.

---

## Configuration

Toute la configuration est centralisée dans `src/config.py` (dataclasses Python).

### Paramètres clés

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `features.cutoff_date` | `"2010-10-01"` | Date séparant passé (features) et futur (label) |
| `features.horizon_days` | `90` | Fenêtre d'observation du churn (en jours) |
| `model.algorithm` | `"random_forest"` | Algorithme : `random_forest` \| `gbt` \| `logistic` |
| `model.num_trees` | `50` | Nombre d'arbres (RF/GBT) |
| `model.max_depth` | `8` | Profondeur max (RF/GBT) |
| `model.use_cv` | `False` | Cross-validation 3 plis |
| `spark.master` | `"local[*]"` | Spark master URL |
| `spark.driver_memory` | `"2g"` | Mémoire driver Spark |
| `monitoring.psi_threshold_warning` | `0.10` | Seuil PSI WARNING |
| `monitoring.psi_threshold_critical` | `0.25` | Seuil PSI CRITICAL |

### Surcharge via variables d'environnement

Les paramètres peuvent être surchargés avec des variables d'env au format `SECTION__KEY` :

```bash
# Utiliser Yarn en production
export SPARK__MASTER=yarn

# Activer MLflow
export MLFLOW__ENABLED=true
export MLFLOW__TRACKING_URI=http://mlflow-server:5000

# Modifier le seuil PSI
export MONITORING__PSI_THRESHOLD_WARNING=0.05
```

---

## Workflow recommandé

```
┌──────────────────────────────────────────────────────────────────┐
│                    WORKFLOW COMPLET                               │
└──────────────────────────────────────────────────────────────────┘

    1. Installation
       pip install -r requirements.txt

    2. Pipeline complet (1 commande)
       python -m cli.pipeline

    3. API REST
       python -m cli.serve
       → http://localhost:8000/docs

    ─────────────────────────────────────────────────────

    OU étape par étape :

    python -m cli.data_download    # ~5-10 min (téléchargement + features)
    python -m cli.train            # ~2-5 min (entraînement)
    python -m cli.predict          # ~1 min (scoring)
    python -m cli.serve            # démarrage API

    ─────────────────────────────────────────────────────

    Ré-entraîner sans re-télécharger :
    python -m cli.pipeline --skip-download

    Scorer avec un nouveau modèle sans recalculer les features :
    python -m cli.pipeline --skip-download --algo gbt --cv

    Surveiller la dérive des données :
    python -m cli.pipeline --skip-download --skip-train
```

---

## Données

**Source :** UCI Machine Learning Repository — [Online Retail II](https://doi.org/10.24432/C5CG6D)

| Propriété | Valeur |
|-----------|--------|
| Période | Décembre 2009 – Décembre 2011 |
| Transactions | ~1 million de lignes |
| Clients | ~5 900 |
| Produits | ~4 600 |
| Pays | 38 |
| Licence | CC BY 4.0 (usage libre) |

**Citation :**
> Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

---

## Dépendances principales

| Package | Rôle |
|---------|------|
| `pyspark` | Traitement distribué + ML (Pipeline, RF, GBT) |
| `pandas` + `openpyxl` | Lecture du fichier Excel UCI |
| `fastapi` + `uvicorn` | API REST asynchrone |
| `pydantic` | Validation des données (requêtes/réponses API) |
| `pytest` | Framework de tests |
| `mlflow` | Tracking d'expériences ML (optionnel) |
| `requests` | Téléchargement HTTP du dataset |
| `yaml` | Chargement de la configuration |
