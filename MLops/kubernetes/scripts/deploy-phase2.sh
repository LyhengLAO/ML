#!/bin/bash
# ============================================
# deploy-phase2.sh
# Ajoute les ressources de la Phase 2
# au projet deja deploye :
# - NetworkPolicy (securite)
# - PodDisruptionBudget (fiabilite)
# - ServiceMonitor (scraping Prometheus)
# - Dashboard Grafana
# ============================================
set -e
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Deploiement Phase 2"
echo "========================================"

echo ""
echo "==> NetworkPolicies (securite reseau)"
kubectl apply -f k8s/14-networkpolicy.yaml

echo ""
echo "==> PodDisruptionBudget (fiabilite)"
kubectl apply -f k8s/15-poddisruptionbudget.yaml

echo ""
echo "==> ServiceMonitor (requis : monitoring deja installe)"
if kubectl get crd servicemonitors.monitoring.coreos.com &>/dev/null; then
    kubectl apply -f k8s/16-servicemonitor.yaml
    kubectl apply -f k8s/17-grafana-dashboard.yaml
    echo "    OK"
else
    echo "    SKIP : kube-prometheus-stack n'est pas installe."
    echo "    Lancer d'abord : ./scripts/install-monitoring.sh"
fi

echo ""
echo "==> Etat"
kubectl get networkpolicy -n ml-demo
echo ""
kubectl get pdb -n ml-demo

echo ""
echo "========================================"
echo "  Phase 2 terminee"
echo "========================================"
