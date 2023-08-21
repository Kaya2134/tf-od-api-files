apt update
apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10
python3.10 --version
apt install python3-pip python-is-python3 -y
apt install python3.10-venv -y
python3.10 -m venv venv
source venv/bin/activate
python ./init.py
