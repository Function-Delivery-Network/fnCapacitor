# Please update your docker login details or Use Environment variables
sudo docker login -u yourusername -p yourpassword
cd miniodbmysql/
ipaddress=$(dig +short myip.opendns.com @resolver1.opendns.com)
sed -i "s/ipaddress/$ipaddress/g" miniodbmysql/handler.py
faas-cli template pull
sudo faas build
sudo faas push