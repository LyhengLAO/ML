# ML Scoring Platform - Projet Kubernetes Local (Phase 1 + Phase 2)

Mini-plateforme Machine Learning qui illustre **les concepts de Kubernetes de la base à la production**, entièrement en local sur votre machine.

## Deux phases

### Phase 1 — Fondations
- FastAPI servant un modèle de scoring
- PostgreSQL (StatefulSet + PVC)
- Redis (Deployment + Service)
- Job d'initialisation SQL
- CronJob de ré-entraînement
- HPA (auto-scaling)
- Ingress pour accéder au navigateur
- UI web avec statistiques temps réel

### Phase 2 — Production-grade
- Métriques Prometheus custom dans le code Python
- Stack Prometheus + Grafana (kube-prometheus-stack)
- Dashboard Grafana préconfiguré
- ServiceMonitor (scraping automatique)
- NetworkPolicies (sécurité réseau)
- PodDisruptionBudget (haute disponibilité)

## Prérequis

- **Docker**
- **Minikube** v1.32+ (avec CNI Calico pour les NetworkPolicies : `--cni=calico`)
- **kubectl** v1.28+
- **Helm** v3+ (pour le monitoring)

## Démarrage rapide (Phase 1)

```bash
# 1. Cluster (avec Calico pour la Phase 2)
minikube start --cpus=2 --memory=4096 --driver=docker --cni=calico
minikube addons enable ingress
minikube addons enable metrics-server

# 2. Construire l'image dans Minikube
eval $(minikube docker-env)
docker build -t ml-api:1.0 ./app

# 3. Déployer la Phase 1
./scripts/deploy.sh

# 4. Accéder à l'UI (dans un autre terminal)
minikube tunnel
echo "127.0.0.1 mldemo.local" | sudo tee -a /etc/hosts
# Ouvrir http://mldemo.local
```

## Passer à la Phase 2

```bash
# 1. Installer Prometheus + Grafana
./scripts/install-monitoring.sh

# 2. Appliquer NetworkPolicies, PDB, ServiceMonitor
./scripts/deploy-phase2.sh

# 3. Ouvrir Grafana
./scripts/access-grafana.sh
# http://localhost:3000 (admin / admin)

# 4. Ouvrir Prometheus
./scripts/access-prometheus.sh
# http://localhost:9090
```

## Guides PDF

- `GUIDE.pdf` — Phase 1 : déploiement et observation (36 pages)
- `GUIDE-PHASE2.pdf` — Phase 2 : monitoring + sécurité (26 pages)

## Structure

```
k8s-project/
├── README.md
├── app/                                  # Code Python
│   ├── Dockerfile
│   ├── requirements.txt                  # Inclut prometheus-client
│   ├── main.py                           # FastAPI + métriques Prometheus
│   └── templates/index.html              # UI web
├── k8s/                                  # Manifestes Kubernetes
│   ├── 00-namespace.yaml                 # [P1] Namespace ml-demo
│   ├── 01-configmap.yaml                 # [P1] Config non sensible
│   ├── 02-secret.yaml                    # [P1] Mot de passe DB
│   ├── 04-postgres-statefulset.yaml      # [P1] PostgreSQL
│   ├── 05-postgres-service.yaml          # [P1] Service headless
│   ├── 06-redis-deployment.yaml          # [P1] Redis cache
│   ├── 07-redis-service.yaml             # [P1] Service Redis
│   ├── 08-init-job.yaml                  # [P1] Init SQL
│   ├── 09-api-deployment.yaml            # [P1] API 3 répliques
│   ├── 10-api-service.yaml               # [P1] Service API
│   ├── 11-api-hpa.yaml                   # [P1] Auto-scaling
│   ├── 12-retrain-cronjob.yaml           # [P1] Ré-entraînement planifié
│   ├── 13-ingress.yaml                   # [P1] Ingress HTTP
│   ├── 14-networkpolicy.yaml             # [P2] Sécurité réseau
│   ├── 15-poddisruptionbudget.yaml       # [P2] Haute dispo
│   ├── 16-servicemonitor.yaml            # [P2] Scrape Prometheus
│   └── 17-grafana-dashboard.yaml         # [P2] Dashboard
└── scripts/
    ├── deploy.sh                         # [P1] Déploie la base
    ├── install-monitoring.sh             # [P2] Helm + Prometheus + Grafana
    ├── deploy-phase2.sh                  # [P2] NetworkPolicy + PDB
    ├── access-grafana.sh                 # [P2] Port-forward Grafana
    ├── access-prometheus.sh              # [P2] Port-forward Prometheus
    ├── load-test.sh                      # Stress test (voir HPA + Grafana)
    ├── observe.sh                        # État du cluster
    └── teardown.sh                       # Nettoyage
```

## Nettoyer

```bash
./scripts/teardown.sh       # Supprime ml-demo et monitoring
minikube stop
minikube delete             # Supprimer complètement
```
