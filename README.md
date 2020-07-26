# Machine Learning Nanodegree - Capstone project

This repository contains my capstone project for [Udacity Machine Learning Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t).

In this project I addressed the problem of classification of high-energy particle collisions into either Signal or Background events in the context of detection of Higgs Boson.

For more details, see the [proposal](proposal/proposal.pdf) and [final report](report/report.pdf)

## Technology stack

In my project I choose to use [Amazon SageMaker](https://aws.amazon.com/sagemaker/) for all aspects related to data exploration/visualization, training and deployment.

The Deep Neural Network is implemented using [PyTorch](https://pytorch.org/).

As part of the project I also implemented a REST API to make predictions. The API is using [Amazon API Gateway](https://aws.amazon.com/api-gateway/) and [AWS Lambda](https://aws.amazon.com/lambda/). For inference, the Lambda function calls the PyTorch model which was deployed as a SageMaker Endpoint.

---
