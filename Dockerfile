FROM python:3
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV DISPLAY :0
#RUN apt-get install -qqy x11-apps
