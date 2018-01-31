# XSOCK=/tmp/.X11-unix/X0
# docker run -it --rm --name mys --volume "$PWD":/usr/src/app -v $XSOCK:$XSOCK mys python3 plotter.py
FROM python:3
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV DISPLAY :0
#RUN apt-get install -qqy x11-apps
