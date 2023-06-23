# syntax=docker/dockerfile:1

FROM zironycho/pytorch:1.6.0-slim-py3.7-v1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY src .