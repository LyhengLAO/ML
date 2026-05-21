import pandas as pd
import shutil
from pathlib import Path
import sys
from typing import Optional
from urllib.request import urlretrieve
import zipfile
import requests

from pyspark.sql import SparkSession, DataFrame

from src.config import DataConfig
from src.data.schema import TRANSACTIONS_SCHEMA

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
root = Path(__file__).resolve().parent.parent
dest_dir = root / "data/raw"
zip_dir = dest_dir / "online_retail_II.zip"
extract_dir = dest_dir / "extracted"
csv_path = dest_dir / "online_retail_II.csv"

def zip_file_download(url = UCI_URL, dest = dest_dir, zip = zip_dir):
    dest.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        download = 0
        with open(zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    download += len(chunk)
                    pct = min(100, download * 100 / total) if total > 0 else 0
                    sys.stdout.write(f"\r  {pct:.1f}%")
                    sys.stdout.flush()

def extract(zip = zip_dir, file = extract_dir):
    file.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip) as z:
        z.extractall(file)
    
    xlsx_files = list(extract_dir.rglob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError("No .xlsx found in the UCI archive.")
    xlsx_path = xlsx_files[0]

    print("Reading Excel sheets....")
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)

    df = df.rename(columns={
        "Invoice":     "InvoiceNo",
        "Price":       "UnitPrice",
        "Customer ID": "CustomerID",
    })
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["CustomerID"] = df["CustomerID"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
    df = df[["InvoiceNo", "StockCode", "Description", "Quantity",
             "InvoiceDate", "UnitPrice", "CustomerID", "Country"]]

    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df):,} rows)")

    # Cleanup
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_dir.unlink(missing_ok=True)

    return csv_path

def load_csv(spark: SparkSession, path: str) -> DataFrame:
    """
    Lit un CSV de transactions avec le schéma canonique.

    Note : on utilise un schéma explicite (plus rapide et plus sûr
    que `inferSchema`) et `mode=DROPMALFORMED` pour ignorer les lignes
    mal formées plutôt que de planter.
    """
    return (
        spark.read
             .schema(TRANSACTIONS_SCHEMA)
             .option("header", True)
             .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
             .option("nullValue", "")
             .option("mode", "DROPMALFORMED")
             .csv(path)
    )


def load_parquet(spark: SparkSession, path: str) -> DataFrame:
    """Lit un Parquet (format préféré pour les sauvegardes intermédiaires)."""
    return spark.read.parquet(path)


def save_parquet(df: DataFrame, path: str, partition_by: Optional[list] = None) -> None:
    """Sauvegarde au format Parquet (rapide, compact, schéma intégré)."""
    writer = df.write.mode("overwrite")
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.parquet(path)


def load_transactions(spark: SparkSession,
                      cfg: Optional[DataConfig] = None,
                      csv_path: Optional[str] = None) -> DataFrame:
    """
    Point d'entrée unique pour charger les transactions.

    Stratégie de résolution :
    1. Si `csv_path` est fourni, on le lit.
    2. Sinon on utilise `cfg.full_csv` (par défaut data/raw/online_retail_II.csv).

    Si le fichier n'existe pas, on lève une erreur claire qui rappelle
    comment le télécharger.
    """
    cfg = cfg or DataConfig()
    target = csv_path or cfg.full_csv

    if not Path(target).exists():
        raise FileNotFoundError(
            f"\n{'=' * 70}\n"
            f"  Le fichier {target} est introuvable.\n"
            f"{'=' * 70}\n"
            f"  Ce projet utilise le dataset open source 'Online Retail II'\n"
            f"  (UCI ML Repository, CC BY 4.0).\n\n"
            f"  Pour le télécharger automatiquement :\n"
            f"      python -m cli.download_dataset\n\n"
            f"  Ou pour utiliser un autre CSV au même schéma :\n"
            f"      python -m cli.train --csv /chemin/vers/fichier.csv\n"
            f"{'=' * 70}"
        )

    print(f">> Source : {target}")
    return load_csv(spark, target)
