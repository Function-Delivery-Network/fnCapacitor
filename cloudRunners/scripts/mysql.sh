faas-cli login --tls-no-verify --username admin --password $(cat password.txt) --gateway http://127.0.0.1:31112
cd mysql-function-openfaas/
kubectl create secret generic secret-mysql-key --from-file=secret-mysql-key=$HOME/secrets/secret_mysql_key.txt --namespace openfaas-fn
faas-cli template pull
sudo faas deploy --gateway http://127.0.0.1:31112