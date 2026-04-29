#!/bin/bash
# Port-forward pour acceder a Prometheus UI
echo "Prometheus sera accessible sur http://localhost:9090"
echo ""
echo "Quelques requetes interessantes a tester :"
echo "  - ml_predictions_total"
echo "  - rate(ml_predictions_total[1m])"
echo "  - histogram_quantile(0.95, rate(ml_prediction_duration_seconds_bucket[5m]))"
echo ""
echo "Ctrl+C pour arreter"
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
