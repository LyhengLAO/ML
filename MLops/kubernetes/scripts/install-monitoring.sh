#!/bin/bash
# ============================================
# install-monitoring.sh
# Installe Prometheus + Grafana (kube-prometheus-stack)
# via Helm. Cree le namespace "monitoring".
# ============================================
set -e

echo "========================================"
echo "  Installation du stack de monitoring"
echo "  (Prometheus + Grafana + Alertmanager)"
echo "========================================"

# Verifier Helm
if ! command -v helm &> /dev/null; then
    echo "Helm non trouve. Installation..."
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

echo ""
echo "==> Ajout du repo Prometheus Community"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

echo ""
echo "==> Installation de kube-prometheus-stack"
echo "    (peut prendre 2-3 minutes...)"

helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --version 58.1.3 \
  --set grafana.adminPassword=admin \
  --set grafana.service.type=ClusterIP \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.resources.requests.memory=256Mi \
  --set prometheus.prometheusSpec.resources.requests.cpu=100m \
  --set alertmanager.enabled=false \
  --wait --timeout=5m

echo ""
echo "==> Etat du monitoring"
kubectl get pods -n monitoring

echo ""
echo "==> Application du ServiceMonitor et du dashboard"
kubectl apply -f "$(dirname "$0")/../k8s/16-servicemonitor.yaml"
kubectl apply -f "$(dirname "$0")/../k8s/17-grafana-dashboard.yaml"

echo ""
echo "========================================"
echo "  Installation terminee !"
echo "========================================"
echo ""
echo "Pour acceder a Grafana :"
echo "   ./scripts/access-grafana.sh"
echo ""
echo "Login : admin / admin"
echo ""
echo "Pour acceder a Prometheus :"
echo "   ./scripts/access-prometheus.sh"
