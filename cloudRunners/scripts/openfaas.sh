#sudo kubectl apply -f kubernetes-metric-server/
sudo kubectl apply -f scripts/01-namespaces.yaml
sleep 20
#sudo kubectl apply -f kube-state-metrics/
#sudo kubectl apply -k cadvisor/
#sudo kubectl apply -f node-exporter/
# Create the operator to instanciate all CRDs
kubectl -n monitoring apply -f ./monitoring/prometheus-operator/

# Deploy monitoring components
kubectl -n monitoring apply -f ./monitoring/node-exporter/
kubectl -n monitoring apply -f ./monitoring/kube-state-metrics/
kubectl -n monitoring apply -f ./monitoring/alertmanager

# Deploy prometheus instance and all the service monitors for targets
kubectl -n monitoring apply -f ./monitoring/prometheus-cluster-monitoring/

# Dashboarding
kubectl -n monitoring create -f ./monitoring/grafana/

# kubectl apply -f metrics-server-exporter/

curl -sL https://cli.openfaas.com | sudo sh

PASSWORD=$(head -c 12 /dev/urandom | shasum| cut -d' ' -f1)

echo $PASSWORD > password.txt

echo $PASSWORD

sudo kubectl -n openfaas create secret generic basic-auth \
--from-literal=basic-auth-user=admin \
--from-literal=basic-auth-password="$PASSWORD"

#sudo cp ../prometheus/prometheus-cfg.yml ../faas-netes/yaml/prometheus-cfg.yml

sudo kubectl apply -f faas-netes/

sleep 60

sudo kubectl port-forward svc/gateway -n openfaas 31112:8080 &

sleep 10

export OPENFAAS_URL=http://127.0.0.1:31112

#echo -n $(cat password.txt) | faas-cli login --password-stdin

#faas-cli store deploy sentimentanalysis
#faas-cli store deploy figlet

#sleep 10


#sleep 10

#faas-cli store deploy colorise

#sleep 10

#faas-cli deploy --image=theaxer/classify:latest --name=classify

#sleep 10

#faas-cli store deploy face-blur

#sleep 10

#faas-cli store deploy face-detect-pigo

#sleep 10
