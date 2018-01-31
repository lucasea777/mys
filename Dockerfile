# docker build -t mys .
# xhost +local:
# docker run -it --rm --name mys --volume "$PWD":/usr/src/app --volume="/tmp/.X11-unix:/tmp/.X11-unix" --env="QT_X11_NO_MITSHM=1" -e DISPLAY=$DISPLAY ipython3 -i plotter.py
# xhost -local:
FROM debian
RUN apt-get update
RUN apt-get install -qqy x11-apps python3 python3-pip python3-setuptools python3-dev
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
ENV DISPLAY :0
RUN apt-get install -qqy python3-tk
#RUN apt-get install -qqy x11-apps
