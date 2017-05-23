# keras-tutorial
Keras Tutorial for Tucson Data Science Meetup

Relevant Keras Documentation [(Sequential Models)](https://keras.io/models/sequential/)
----------------------------

  * [The Model class](https://keras.io/models/model/)
  * [Commonly used layers](https://keras.io/layers/core/)
  * [Optimizers](https://keras.io/optimizers/)

A sequential model is just a network that consists of a series of layers ([input]->[hidden #1]->...->[output]), the word *sequential* has nothing to do with RNN's.
What's cool about sequential models?
  * No need to declare weights or biases
  * Only need to declare number of nodes in the layer and the activation type
  * All of the most commonly used layer types (fully-connected, convolutional, recurrent, etc.) are just a callback away
  * Much easier to implement than in TensorFlow
