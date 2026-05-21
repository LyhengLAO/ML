"""
Lance le serveur FastAPI de prédiction de churn.

Ce script configure les variables d'environnement (APP_ENV, MODEL_DIR)
avant de démarrer uvicorn, ce qui permet à api/dependencies.py de les lire
au moment du startup (lifespan FastAPI).

Usage :
    python -m cli.serve
    python -m cli.serve --port 8080
    python -m cli.serve --model-dir output/model --env production
    python -m cli.serve --reload         # développement (hot-reload)
    python -m cli.serve --workers 4      # production multi-process

Workflow complet :
    python -m cli.data_download          # 1. prépare les données
    python -m cli.train                  # 2. entraîne le modèle
    python -m cli.serve                  # 3. lance l'API

Documentation interactive : http://localhost:8000/docs
"""
import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Serveur FastAPI — Churn Prediction API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--host", default="0.0.0.0",
                   help="Adresse d'écoute. Défaut: 0.0.0.0 (toutes interfaces).")
    p.add_argument("--port", type=int, default=8000,
                   help="Port d'écoute. Défaut: 8000.")
    p.add_argument("--env", default="local",
                   help="Environnement de configuration (local | production). Défaut: local.")
    p.add_argument("--model-dir", default=None,
                   help="Chemin du PipelineModel entraîné. Défaut: output/model.")
    p.add_argument("--reload", action="store_true",
                   help="Mode développement avec hot-reload (désactive --workers).")
    p.add_argument("--workers", type=int, default=1,
                   help="Nombre de workers uvicorn (prod). Défaut: 1. Ignoré si --reload.")
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"],
                   help="Niveau de log uvicorn. Défaut: info.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Variables d'environnement (lues par api/dependencies.py au startup) ──
    os.environ["APP_ENV"] = args.env
    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    # ── Vérification pre-flight ────────────────────────────────────────────────
    model_dir = args.model_dir or os.getenv("MODEL_DIR", "output/model")
    from pathlib import Path
    if not Path(model_dir).exists():
        print(f"\n  ⚠️  Modèle introuvable : {model_dir}")
        print("  L'API démarrera en mode dégradé.")
        print("  Entraînez d'abord : python -m cli.train\n")

    # ── Bannière ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Churn Prediction API")
    print("=" * 60)
    print(f"  Environnement : {args.env}")
    print(f"  Modèle        : {model_dir}")
    print(f"  URL           : http://{args.host}:{args.port}")
    print(f"  Docs          : http://{args.host}:{args.port}/docs")
    print(f"  Mode          : {'développement (reload)' if args.reload else 'production'}")
    print(f"  Workers       : {1 if args.reload else args.workers}")
    print("=" * 60)
    print()

    # ── Uvicorn ───────────────────────────────────────────────────────────────
    try:
        import uvicorn
    except ImportError:
        print("ERREUR : uvicorn n'est pas installé.")
        print("  pip install uvicorn[standard]")
        sys.exit(1)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
