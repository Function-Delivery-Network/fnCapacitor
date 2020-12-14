# Kubernetes 1.18.4 Monitoring Guide

Create a cluster with [kind](https://kind.sigs.k8s.io/docs/user/quick-start/)
```
kind create cluster --name prometheus --image kindest/node:v1.18.4
```

```
kubectl create ns monitoring

# Create the operator to instanciate all CRDs
kubectl -n monitoring apply -f ./prometheus-monitoring/kubernetes/1.18.4/prometheus-operator/

# Deploy monitoring components
kubectl -n monitoring apply -f ./prometheus-monitoring/kubernetes/1.18.4/node-exporter/
kubectl -n monitoring apply -f ./prometheus-monitoring/kubernetes/1.18.4/kube-state-metrics/
kubectl -n monitoring apply -f ./prometheus-monitoring/kubernetes/1.18.4/alertmanager

# Deploy prometheus instance and all the service monitors for targets
kubectl -n monitoring apply -f ./prometheus-monitoring/kubernetes/1.18.4/prometheus-cluster-monitoring/

# Dashboarding
kubectl -n monitoring create -f ./prometheus-monitoring/kubernetes/1.18.4/grafana/

```