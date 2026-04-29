#!/bin/bash
# Port-forward pour acceder a Grafana
# Login : admin / admin
echo "Grafana sera accessible sur http://localhost:3000"
echo "Login : admin / admin"
echo "Dashboard : chercher 'ML Scoring API' dans Dashboards"
echo ""
echo "Ctrl+C pour arreter"
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
