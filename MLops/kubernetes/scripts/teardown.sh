#!/bin/bash
# ============================================
# teardown.sh - Supprime tout le projet
# ============================================
set +e

echo "Suppression du namespace ml-demo..."
kubectl delete namespace ml-demo --ignore-not-found=true

echo "Suppression du monitoring..."
helm uninstall monitoring -n monitoring 2>/dev/null || true
kubectl delete namespace monitoring --ignore-not-found=true

echo ""
echo "Termine. Le cluster Minikube reste actif."
echo "Pour l'arreter : minikube stop"
echo "Pour le supprimer : minikube delete"
