# What's On Pizza Project

What's On Pizza - project that give you an ability to know pizza name by photo.
This repo contains server side source code and similiar training/validation scripts using MXNet and TensorFlow.
You can compare two frameworks that solve one task. 
Client side can be found [here](https://github.com/Lohika-Labs/whatsonpizza-mobile).

## Task

Build an ML model that would look at a picture of a pizza and output a list of possible pizza names.
We trained a model for a single-label classification, where the output just the name of the pizza.
Currently only ten pizza names

## Dataset

Pizza images were collected with [pizza scrapper](https://github.com/Lohika-Labs/whatsonpizza/tree/master/pizza_scraper) (two python scripts for parsing pdf and html, examples of collected images can be found inside [taxonomy](https://github.com/Lohika-Labs/whatsonpizza/tree/master/taxonomy/images) folder).
[Classifier](https://github.com/Lohika-Labs/whatsonpizza/tree/master/whatsonpizza_classifier) used for sorting pizza images and cleaning dataset from unrepresentative data.
Trained models are used on the backend, which logic can be found [here](https://github.com/Lohika-Labs/whatsonpizza/tree/master/whatsonpizza_backend). It can be easily deployed with Docker.

## The code

We used [Inception-BN model](https://github.com/dmlc/mxnet-model-gallery/blob/master/imagenet-1k-inception-bn.md) as a pre-trained model and fine-tune as a technique for changing initial model wights using collected data.
* [TensorFlow fine-tunning](https://github.com/Lohika-Labs/whatsonpizza/blob/master/tf_dev/inception_retrain.py)
* [MXNet fine-tunning](https://github.com/Lohika-Labs/whatsonpizza/blob/master/finetune.py)

After training put output models at [models](https://github.com/Lohika-Labs/whatsonpizza/tree/master/whatsonpizza_backend/models) folder otherwise backend will not be able to work.

## Classification
* [MXNet classifier](https://github.com/Lohika-Labs/whatsonpizza/blob/master/whatsonpizza_backend/backend/mxclassifier.py)
* [TensorFlow classifier](https://github.com/Lohika-Labs/whatsonpizza/blob/master/whatsonpizza_backend/backend/tfclassifier.py)

For testing MXNet model without backend deployment please use [testmodels.py](https://github.com/Lohika-Labs/whatsonpizza/blob/master/testmodels.py) or [demo.py](https://github.com/Lohika-Labs/whatsonpizza/blob/master/demo.py).

Please, write us for more details dmaiboroda@lohika.com
