Creat client machine using ubuntu 20.04<br/>

#sudo apt -y update; sudo apt upgrade;<br/>
#sudo apt install python3-openstackclient<br/>
#sudo snap install openstackclients<br/>
#sudo apt install python3-openstackclient<br/>
#sudo apt install python3-novaclient<br/>
#sudo apt install python3-keystoneclient<br/>
#sudo apt update<br/>
#sudo apt install python3-pip<br/>
------- setup ssh-keys in ...-for  cloud-cfg.txt<br/>
ubuntu<br/>
#mkdir -p /home/ubuntu/cluster-keys<br/>
#ssh-keygen -t rsa<br/>
Set the file path /home/ubuntu/cluster-keys/cluster-key<br/>
<br/>
<br/>
------- clone the github and create vms<br/>
#git clone https://github.com/resa-git/model_serving<br/>
<br/>
copy ssh key from cluster-keys.pub to prod-cloud-cfg.txt<br/><br/>
#cp  prod-cloud-cfg.txt  dev-cloud-cfg.txt<br/>
#cp  prod-cloud-cfg.txt  w1-cloud-cfg.txt<br/>
#cp  prod-cloud-cfg.txt  w2-cloud-cfg.txt<br/>
<br/>
copy openrc.sh file from openstack api to this vm<br/>
#scp -i .ssh/file.pem openrc.sh ubuntu@floatingIp:~<br/>
#source openrc.sh<br/>
<br/>
Edit information of start_instances.py<br/>
#python3 start_instances.py<br/>
<br/>
-------------------------- ansible ----------<br/>
#sudo python3 -m pip install ansible<br/>
Edit web.ini and add the ip of the nodes<br/>
---------------------- run ansible --------<br/>
#ansible-playbook -i web.ini configuration.yml<br/>
<br/>
<br/>
Please note that we have sometimes problem with pulling the image from dockerhub production_server_web:latest and production_server_worker_1:latest, in this case we need to login manually into prod server and run<br/>
#sudo docker stack deploy -c stack1.yaml stackdemo<br/>


-------------- development server --------------<br/>
To change the ML program, login to the devserver and go to the /model_serving/ci_cd/development_server/ and edit and run tuning_no_ray.py, then go to the ~/jump and run<br/>
#git add final.sav<br/>
#git commit <br/>
#git push production master<br/>
Then the modified model is on the production server.<br/>

