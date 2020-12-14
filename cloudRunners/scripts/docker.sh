sudo apt-get -y update

sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     gnupg-agent \
     software-properties-common -y

sleep 1

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

sleep 1

sudo apt-key fingerprint 0EBFCD88

sleep 1

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

sleep 1

sudo apt-get update -y

apt-get install -y \
    containerd.io=1.2.10-3 \
    docker-ce=5:19.03.4~3-0~ubuntu-$(lsb_release -cs) \
    docker-ce-cli=5:19.03.4~3-0~ubuntu-$(lsb_release -cs)