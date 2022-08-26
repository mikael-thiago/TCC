FROM python:3.9.13-alpine3.16

RUN pip install tensorflow
RUN pip install keras
RUN pip install gym
RUN pip install ale-py

WORKDIR /a3c

COPY . /a3c

RUN ale-import-roms roms/

CMD python train.py