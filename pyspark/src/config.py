"""
Gestion centralisée de la configuration.

Charge la config depuis :
1. Un fichier YAML (configs/{env}.yaml)
2. Les variables d'environnement (qui ont priorité)

Usage :
    from churn_pipeline.config import load_config
    cfg = load_config("local")  # ou "production"
    print(cfg.data.raw_path)
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Racine du projet (3 niveaux au-dessus de ce fichier)
PROJECT_ROOT = Path(__file__).resolve().parents[2].parent


@dataclass
class SparkConfig:
    app_name: str = "ChurnPipeline"
    master: str = "local[*]"
    shuffle_partitions: int = 8
    driver_memory: str = "2g"
    executor_memory: str = "4g"
    adaptive_enabled: bool = True


@dataclass
class DataConfig:
    raw_path: str = "data/raw"
    interim_path: str = "data/interim"
    processed_path: str = "data/processed"
    external_path: str = "data/external"
    sample_csv: str = "data/raw/sample_transactions.csv"
    full_csv: str = "data/raw/online_retail_II.csv"


@dataclass
class FeaturesConfig:
    cutoff_date: str = "2010-10-01"
    horizon_days: int = 90
    feature_cols: list = field(default_factory=lambda: [
        "recency", "frequency", "monetary",
        "avg_basket", "n_products", "n_countries", "tenure_days",
    ])


@dataclass
class ModelConfig:
    algorithm: str = "random_forest"  # random_forest | gbt | logistic
    num_trees: int = 50
    max_depth: int = 8
    min_instances_per_node: int = 5
    train_ratio: float = 0.8
    seed: int = 42
    use_cv: bool = False


@dataclass
class MLflowConfig:
    enabled: bool = False
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "churn_prediction"
    registered_model_name: str = "churn_model"


@dataclass
class MonitoringConfig:
    psi_threshold_warning: float = 0.10
    psi_threshold_critical: float = 0.25


@dataclass
class Config:
    env: str = "local"
    output_dir: str = "output"
    spark: SparkConfig = field(default_factory=SparkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


def _merge_dict(base: dict, override: dict) -> dict:
    """Fusion récursive de deux dicts (override gagne)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(cfg_dict: dict) -> dict:
    """Surcharge depuis les variables d'environnement.

    Convention : SECTION__KEY (ex. SPARK__MASTER=yarn ou MLFLOW__ENABLED=true).
    Les vars d'environnement ont la priorité absolue, ce qui permet de
    surcharger n'importe quel paramètre depuis un Dockerfile, K8s ConfigMap, etc.
    """
    # Liste des sections valides (pour éviter de polluer avec des vars random)
    valid_sections = {"spark", "data", "features", "model", "mlflow", "monitoring"}

    for env_key, value in os.environ.items():
        if "__" not in env_key:
            continue
        section, key = env_key.lower().split("__", 1)
        if section not in valid_sections:
            continue
        cfg_dict.setdefault(section, {})
        # Cast simple : bool, int, float, sinon string
        if value.lower() in ("true", "false"):
            cast_value = value.lower() == "true"
        else:
            try:
                cast_value = int(value)
            except ValueError:
                try:
                    cast_value = float(value)
                except ValueError:
                    cast_value = value
        cfg_dict[section][key] = cast_value
    return cfg_dict


def _dict_to_config(d: dict) -> Config:
    """Convertit un dict en objet Config typé."""
    return Config(
        env=d.get("env", "local"),
        output_dir=d.get("output_dir", "output"),
        spark=SparkConfig(**d.get("spark", {})),
        data=DataConfig(**d.get("data", {})),
        features=FeaturesConfig(**d.get("features", {})),
        model=ModelConfig(**d.get("model", {})),
        mlflow=MLflowConfig(**d.get("mlflow", {})),
        monitoring=MonitoringConfig(**d.get("monitoring", {})),
    )


def load_config(env: Optional[str] = None,
                config_dir: Optional[Path] = None) -> Config:
    """
    Charge la config pour un environnement donné.

    Ordre de précédence :
    1. configs/default.yaml (base)
    2. configs/{env}.yaml (overrides)
    3. variables d'environnement (priorité max)
    """
    env = env or os.getenv("APP_ENV", "local")
    config_dir = config_dir or (PROJECT_ROOT / "configs")

    base = _load_yaml(config_dir / "default.yaml")
    env_specific = _load_yaml(config_dir / f"{env}.yaml")

    merged = _merge_dict(base, env_specific)
    merged = _apply_env_overrides(merged)
    merged.setdefault("env", env)

    return _dict_to_config(merged)
