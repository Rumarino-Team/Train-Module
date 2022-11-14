# Usage: pip install -r requirements.txt


# Dowload the yolov7 repository
git clone https://github.com/WongKinYiu/yolov7.git



# Intall Pip for dowloading python packages
py -m pip install --upgrade pip
py -m pip --version

#  Install virtualenv packages for making the virtualenviroment for not having problems with
#  the global python packages
pip install virtualenv

#  Create a virtual enviroment
virtualenv venv

#  Activate the virtual enviroment
venv\Scripts\activate

#  Install the python dependencies
pip install -r yolov7/requirements.txt

