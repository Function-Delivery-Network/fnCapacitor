# Please update your docker login details or Use Environment variables
sudo docker login -u yourusername -p yourpassword
cd queryfunction/
ipaddress=$(dig +short myip.opendns.com @resolver1.opendns.com)
sed -i "s/ipaddress/$ipaddress/g" queryfunction/handler.py
faas-cli template pull
sudo faas build
sudo faas push