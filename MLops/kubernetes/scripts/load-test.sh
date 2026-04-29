#!/bin/bash
# ============================================
# load-test.sh - Stress test l'API pour declencher le HPA
# ============================================
set -e

NS="ml-demo"
DUREE=${1:-180}   # Duree par defaut : 3 minutes
CLIENTS=${2:-50}  # Clients concurrent

echo "========================================"
echo "  Stress test de l'API"
echo "========================================"
echo "  Duree : ${DUREE}s"
echo "  Clients concurrent : ${CLIENTS}"
echo ""
echo "Suivre le HPA dans un autre terminal :"
echo "  kubectl get hpa -n $NS -w"
echo ""
echo "Suivre les pods :"
echo "  kubectl get pods -n $NS -w"
echo ""
echo "Demarrage dans 3 secondes..."
sleep 3

# Lance un pod qui genere la charge depuis l'interieur du cluster
kubectl run load-test \
  --rm -it \
  --image=alpine/curl:latest \
  --restart=Never \
  -n $NS \
  --command -- sh -c "
apk add --no-cache coreutils > /dev/null 2>&1
END=\$(( \$(date +%s) + ${DUREE} ))
COUNT=0
while [ \$(date +%s) -lt \$END ]; do
  for i in \$(seq 1 ${CLIENTS}); do
    curl -s -X POST http://ml-api/predict \
      -H 'Content-Type: application/json' \
      -d '{\"age\":35,\"revenu\":50000,\"anciennete\":5,\"dettes\":1000}' \
      -o /dev/null &
  done
  wait
  COUNT=\$((COUNT + ${CLIENTS}))
  echo \"Requetes envoyees : \$COUNT\"
done
echo \"Test termine. Total : \$COUNT requetes.\"
"
