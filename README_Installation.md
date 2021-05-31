Creat client machine using ubuntu 20.04

#sudo apt -y update; sudo apt upgrade;
#sudo apt install python3-openstackclient
#sudo snap install openstackclients
#sudo apt install python3-openstackclient
#sudo apt install python3-novaclient
#sudo apt install python3-keystoneclient
#sudo apt update
#sudo apt install python3-pip
------- setup ssh-keys in ...-for  cloud-cfg.txt
ubuntu
#mkdir -p /home/ubuntu/cluster-keys
#ssh-keygen -t rsa
Set the file path /home/ubuntu/cluster-keys/cluster-key


------- clone the github and create vms
#git clone https://github.com/resa-git/model_serving

copy ssh key from cluster-keys.pub to prod-cloud-cfg.txt
#cp  prod-cloud-cfg.txt  dev-cloud-cfg.txt
#cp  prod-cloud-cfg.txt  w1-cloud-cfg.txt
#cp  prod-cloud-cfg.txt  w2-cloud-cfg.txt

copy openrc.sh file from openstack api to this vm
#scp -i .ssh/file.pem openrc.sh ubuntu@floatingIp:~
#source openrc.sh

Edit information of start_instances.py
#python3 start_instances.py

-------------------------- ansible ----------
#sudo python3 -m pip install ansible
Edit web.ini and add the ip of the nodes
---------------------- run ansible --------
#ansible-playbook -i web.ini configuration.yml


Please note that we have sometimes problem with pulling the image from dockerhub web_image:latest and workers:latest, in this case we need to login manually into prod server and run
#sudo docker stack deploy -c stack1.yaml stackdemo
