variable "image" { type = string }
variable "region" { type = string }
variable "instance" {type = string}
variable "master_nodes" { type = list }
variable "worker_nodes" { type = list }

provider "aws" {
  profile = "default"
  region  = var.region
}

resource "aws_default_vpc" "default" {
  tags = {
    Name = "Default VPC"
  }
}

resource "aws_subnet" "private_subnet" {
    cidr_block = "172.31.89.0/24"
    vpc_id = aws_default_vpc.default.id
  tags = {
    Name = "Private Subnet for Kubernetes Cluster"
  }
}

resource "aws_key_pair" "key" {
  key_name   = "same_for_all"
  public_key = file("~/.ssh/id_rsa.pub")
}

resource "aws_instance" "Worker" {
  count         = length(var.worker_nodes) 
  ami           = var.image
  instance_type = var.instance
  tags = {
    Name = var.worker_nodes[count.index]
  }
  key_name      = "same_for_all"
  associate_public_ip_address   = true
  subnet_id     = aws_subnet.private_subnet.id
}

resource "aws_instance" "Master" {
  count         = length(var.master_nodes) 
  ami           = var.image
  instance_type = var.instance
  associate_public_ip_address = true
  key_name = "same_for_all"
  subnet_id     = aws_subnet.private_subnet.id
  #private_ip = var.master_nodes_private_ip[count.index]
  provisioner "local-exec" {
    command = "cd ../cloudRunners && tar -zcvf ../folders.tar.gz $(ls -d */)"
  }
  provisioner "file" {
    source      = "~/.ssh/id_rsa"
    destination = "/home/ubuntu/id_rsa"
  }
  provisioner "file" {
    source      = "../folders.tar.gz"
    destination = "/home/ubuntu/folders.tar.gz"
  }
  connection {
    type        = "ssh"
    user        = "ubuntu"
    private_key = file("~/.ssh/id_rsa")
    host        = aws_instance.Master[count.index].public_ip
  }
  provisioner "remote-exec" {
    inline = [
      "sudo apt-get -y update",
      "sudo bash -c 'echo 1 > /proc/sys/net/ipv4/ip_forward'",
      "cd /home/ubuntu",
      "tar -zxvf folders.tar.gz",
      "rm -rf *.tar.gz",
      "sudo chmod 600 id_rsa",
      "sudo sh ./scripts/docker.sh",
      "sudo sh ./scripts/kubeadm.sh",
      "scp -r -oStrictHostKeyChecking=no -i id_rsa ./scripts ubuntu@${aws_instance.Worker[0].private_ip}:/home/ubuntu/",
      "scp -r -oStrictHostKeyChecking=no -i id_rsa ./scripts ubuntu@${aws_instance.Worker[1].private_ip}:/home/ubuntu/",
      "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[0].private_ip} 'sudo sh scripts/docker.sh && sudo sh scripts/kubeadm.sh'",
      "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[1].private_ip} 'sudo sh scripts/docker.sh && sudo sh scripts/kubeadm.sh'",
      "sudo kubeadm init",
      "mkdir -p $HOME/.kube",
      "sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config",
      "sudo chown $(id -u):$(id -g) $HOME/.kube/config",
      "sudo sh ./scripts/weave.sh",
      "sudo kubeadm token create --print-join-command | grep 'kubeadm join' | grep 'kubeadm join' > join.sh",
      "scp -oStrictHostKeyChecking=no -i id_rsa join.sh ubuntu@${aws_instance.Worker[0].private_ip}:/home/ubuntu/join.sh",
      "scp -oStrictHostKeyChecking=no -i id_rsa join.sh ubuntu@${aws_instance.Worker[1].private_ip}:/home/ubuntu/join.sh",
      "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[0].private_ip} 'sudo sh join.sh ; '",
      "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[1].private_ip} 'sudo sh join.sh ; '",
      "echo 'nameserver 10.96.0.10' > sudo /etc/resolv.conf",
      "sudo sh ./scripts/openfaas.sh"
    ]
  }
  provisioner "local-exec" {
    command = "cd .. && rm -rf *.tar.gz"
  }
  tags = {
    Name = var.master_nodes[count.index]
  }
}

output "master_ip" {
  value = aws_instance.Master[0].public_ip
}
// "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[0].private_ip} 'sudo sh join.sh && kubectl apply -f kubernetes-metric-server/ ; '",
      // "ssh -oStrictHostKeyChecking=no -i id_rsa ubuntu@${aws_instance.Worker[1].private_ip} 'sudo sh join.sh && kubectl apply -f kubernetes-metric-server/ ; '",
      
