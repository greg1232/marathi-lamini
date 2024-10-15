# Base container name
ARG BASE_NAME=python:3.11

FROM $BASE_NAME as base

ARG PACKAGE_NAME="marathi-llm"

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt

RUN pip install -r requirements.txt

# Copy all files to the container
COPY scripts /app/${PACKAGE_NAME}/scripts
COPY 01_eval /app/${PACKAGE_NAME}/01_eval
COPY 02_data /app/${PACKAGE_NAME}/02_data
#COPY 03_visualize /app/${PACKAGE_NAME}/03_visualize
#COPY 04_train /app/${PACKAGE_NAME}/04_train

WORKDIR /app/${PACKAGE_NAME}


