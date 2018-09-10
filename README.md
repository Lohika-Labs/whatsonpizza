# What's On Pizza Project

What's On Pizza - project that give you an ability to know pizza name by photo.
This repo contains server side source code and similiar training/validation scripts using MxNet and TensorFlow.
You can compare two frameworks that solve one task. 
Client side can be found [here] (https://github.com/Lohika-Labs/whatsonpizza-mobile).

## Task

Build an ML model that would look at a picture of a pizza and output a list of possible pizza names.
So far, the model doesn't perform well, most likely due to a low-quality dataset.
We trained a model for a single-label classification, where the output just the name of the pizza.

## Dataset

Pizza images were collected with [pizza scrapper](https://github.com/Lohika-Labs/whatsonpizza/tree/master/pizza_scraper) (two python scripts for parsing pdf and html, examples of collected images can be found inside [taxonomy](https://github.com/Lohika-Labs/whatsonpizza/tree/master/taxonomy/images) folder).
[Classifier](https://github.com/Lohika-Labs/whatsonpizza/tree/master/whatsonpizza_classifier) used for sorting pizza images and cleaning dataset from unrepresentative data.
Trained models are used on the backend, which logic can be found [here](https://github.com/Lohika-Labs/whatsonpizza/tree/master/whatsonpizza_backend). It can be easily deployed with Docker.

## The code

//TODO

## Evaluation
//TODO
