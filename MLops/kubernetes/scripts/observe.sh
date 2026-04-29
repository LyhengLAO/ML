#!/bin/bash
# ============================================
# observe.sh - Commandes pour observer le cluster
# ============================================
NS="ml-demo"

echo "========================================"
echo "  Vue d'ensemble du namespace"
echo "========================================"
kubectl get all -n $NS

echo ""
echo "========================================"
echo "  Pods en detail"
echo "========================================"
kubectl get pods -n $NS -o wide

echo ""
echo "========================================"
echo "  Consommation de ressources"
echo "========================================"
kubectl top pods -n $NS 2>/dev/null || echo "(metrics-server pas pret)"

echo ""
echo "========================================"
echo "  Etat du HPA"
echo "========================================"
kubectl get hpa -n $NS

echo ""
echo "========================================"
echo "  Jobs et CronJobs"
echo "========================================"
kubectl get jobs,cronjobs -n $NS

echo ""
echo "========================================"
echo "  PVCs"
echo "========================================"
kubectl get pvc -n $NS

echo ""
echo "========================================"
echo "  Events recents"
echo "========================================"
kubectl get events -n $NS --sort-by='.lastTimestamp' | tail -10

echo ""
echo "========================================"
echo "  Services et Ingress"
echo "========================================"
kubectl get svc,ingress -n $NS
