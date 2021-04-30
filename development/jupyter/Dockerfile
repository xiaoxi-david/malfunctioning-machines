FROM python:3.7-slim-buster

RUN apt-get update
RUN apt-get install -y libsndfile1

RUN pip install --upgrade pip
RUN pip install jupyter \
    pandas \
    matplotlib \
    wordcloud \
    ipykernel \
    tensorflow \
    tensorflow-datasets \
    pydub \
    librosa \
    kapre

WORKDIR /notebooks