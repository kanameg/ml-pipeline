ARG VER="3.8"
FROM python:${VER}-slim

RUN apt-get update && apt-get install -y tzdata \
&& apt-get install -y libopencv-dev \
&& apt-get install -y git \
&& apt-get install -y wget \
&& apt-get install -y unzip \
&& apt-get install -y fontconfig \
&& apt-get -y install gcc \
&& rm -rf /var/lib/apt/lists/*
RUN wget https://moji.or.jp/wp-content/ipafont/IPAexfont/IPAexfont00301.zip \
&& unzip IPAexfont00301.zip \
&& mkdir -p /usr/share/fonts/ipa \
&& cp IPAexfont00301/*.ttf /usr/share/fonts/ipa \
&& rm -rf IPAexfont \
&& rm IPAexfont00301.zip \
&& fc-cache -fv

WORKDIR /tmp/work
COPY requirements.txt ${PWD}
COPY kaggle.json /root/.kaggle/
RUN pip install -r requirements.txt

EXPOSE 8888

ENV TZ=Asia/Tokyo