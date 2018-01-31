_Usando docker_
 Solo es necesario correr las siguientes lineas:
 
 $ docker build -t mys .
 $ xhost +local:
 $ docker run -it --rm --name mys \
 --volume "$PWD":/usr/src/app  \ 
 --volume="/tmp/.X11-unix:/tmp/.X11-unix" \ 
 --env="QT_X11_NO_MITSHM=1" -e DISPLAY=$DISPLAY \ 
 ipython3 -i plotter.py
 $ xhost -local:

_Dependencias_

$ sudo apt-get -y --force-yes install python3-tk

$ sudo pip3 install scipy mpmath matplotlib numpy sympy python3-tk ipython

_Correr grafiquito_

$ ipython3 -i plotter.py

- g4ej5() (por ejemplo)
