# Anomaly sound detection in pumps

## Introduction

This repo is my attempt to solve the task [Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring](http://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds) from the [2020 DCASE Challenge](http://dcase.community/challenge2020/) for pumps.

The repo has several [Jupyter notebooks](https://jupyter.org) in the *development* folder and a [Streamlit](https://streamlit.io) webpage in the *production* folder.

## Problem

**Anomalous sound detection (ASD)** is the task to identify whether the sound emitted from a target machine is normal or anomalous.

Anomaly detection techniques can be categorized as:

- **Supervised anomaly detection** requires the entire dataset to be labeled "normal" or "abnormal". This technique is a binary classification task.
- **Semi-supervised anomaly detection** requires only data considered "normal" to be labeled. In this technique, the model will learn what "normal" data are like.
- **Unsupervised anomaly detection** involves unlabeled data. In this technique, the model will learn which data is "normal" and "abnormal".

Automatically detecting mechanical failure is an essential technology in the fourth industrial revolution. Prompt detection of machine anomaly by observing its sounds may be useful for machine condition monitoring.

## Dataset

In 2019, Hitachi released the [MIMII dataset](https://zenodo.org/record/3384388) and one year later, the second task of the 2020 DCASE Challenge used a simplified version of the MIMII dataset.

For this project, I use only the [development dataset](https://zenodo.org/record/3678171) for pumps from the 2020 DCASE Challenge. It contains 3349 audio files for training and 856 audio files for testing. Each recording is a single-channel 10-sec length audio that includes both a target machine's operating sound and environmental noise.

## Methodology

Although anomaly detection problems can be solved with supervised, semi-supervised and unsupervised anomaly detectors, the dataset that I used is only suitable for semi-supervised anomaly detectors.

 ![Anomaly detector](http://d33wubrfki0l68.cloudfront.net/268bbc4666654d6e5ef28c449067626fbfee7488/2ad7c/images/tasks/challenge2020/task2_unsupervised_detection_of_anomalous_sounds_for_machine_condition_monitoring_01.png)

I use [Tensorflow](https://www.tensorflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/) and [Tensorflow serving](https://www.tensorflow.org/tfx/guide/serving) to solve the problem.

## Instructions

To run the code, you need [Docker](https://www.docker.com/products/docker-desktop) and [Docker Compose](https://docs.docker.com/compose/install/). There are two docker compose files.

- The docker compose file in the development folder will launch two containers: one for Jupyter and another for Tensorboard.
- The docker compose file in the production folder wiil launch two containers: one for the Streamlit app and another for Tensorflow serving.

You can run the docker compose files with [VS Code](https://code.visualstudio.com) and the [Docker extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) or with the command *docker compose up*. For instance:
> docker compose up docker-compose.yaml

After you launch the containers, Docker will show the address for the Jupyter notebooks and the Streamlit web.

**NOTE:** The Streamlit app needs some files. You can run the notebook *Files-for-Streamlit-app* to create them or download them from [here](https://drive.google.com/file/d/1kCGUZY6ZG9asS1vIIKR88OTFa-RSZueZ/view?usp=sharing). 

If you download them, you need to unzip the file you download in the frontend folder (production/frontend) to create a folder (store) that has three folders inside (audios, images, json).
