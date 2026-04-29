#!/bin/bash
# ============================================
# deploy.sh - Deploie le projet en respectant l'ordre
# ============================================
set -e

NS="ml-demo"
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Deploiement du projet ML Kubernetes"
echo "========================================"

echo ""
echo "==> 1/10 Namespace"
kubectl apply -f k8s/00-namespace.yaml

echo ""
echo "==> 2/10 ConfigMap et Secret"
kubectl apply -f k8s/01-configmap.yaml
kubectl apply -f k8s/02-secret.yaml

echo ""
echo "==> 3/10 PostgreSQL (StatefulSet + Service)"
kubectl apply -f k8s/05-postgres-service.yaml
kubectl apply -f k8s/04-postgres-statefulset.yaml

echo "    Attente de Postgres..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NS --timeout=180s

echo ""
echo "==> 4/10 Redis (Deployment + Service)"
kubectl apply -f k8s/06-redis-deployment.yaml
kubectl apply -f k8s/07-redis-service.yaml
kubectl wait --for=condition=ready pod -l app=redis -n $NS --timeout=60s

echo ""
echo "==> 5/10 Job d'initialisation DB"
kubectl apply -f k8s/08-init-job.yaml
echo "    Attente de la fin du Job..."
kubectl wait --for=condition=complete job/db-init -n $NS --timeout=120s

echo ""
echo "==> 6/10 API Deployment"
kubectl apply -f k8s/09-api-deployment.yaml
kubectl apply -f k8s/10-api-service.yaml

echo "    Attente que l'API soit prete..."
kubectl wait --for=condition=available deploy/ml-api -n $NS --timeout=180s

echo ""
echo "==> 7/10 HorizontalPodAutoscaler"
kubectl apply -f k8s/11-api-hpa.yaml

echo ""
echo "==> 8/10 CronJob de re-entrainement"
kubectl apply -f k8s/12-retrain-cronjob.yaml

echo ""
echo "==> 9/10 Ingress"
kubectl apply -f k8s/13-ingress.yaml

echo ""
echo "==> 10/10 Etat final"
kubectl get all -n $NS

echo ""
echo "========================================"
echo "  Deploiement termine avec succes !"
echo "========================================"
echo ""
echo "Prochaines etapes :"
echo ""
echo "1. Dans un autre terminal, lancer :"
echo "   minikube tunnel"
echo ""
echo "2. Ajouter dans /etc/hosts :"
echo "   127.0.0.1 mldemo.local"
echo ""
echo "3. Ouvrir http://mldemo.local dans votre navigateur"
echo ""
echo "Ou bien sans Ingress, utiliser un port-forward :"
echo "   kubectl port-forward -n $NS svc/ml-api 8000:80"
echo "   Puis http://localhost:8000"
