# tensorflow-node-examples
[Tensorflow.js](https://github.com/tensorflow/tfjs) Node.js Examples. Inspired from official [tfjs-examples](https://github.com/tensorflow/tfjs-examples).

## Disclaimer
Tensorflow.js for Node.js (namely `tfjs-node`) is currently in development status, see [here](https://github.com/tensorflow/tfjs-node).

## Why?
Because officiale Tensorflow.js examples are mostly written in TypeScript or use the [async/await ES2017](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Statements/async_function) paradigma, while the most of node modules are not.

## Node.js Examples

## Getting Started
Tf.js Hello World for Node.js: training and predict.

## Sentiment
Load and predict sentiment from a pre-trained sentiment model with `tf.loadModel`.

## LSTM
A simple LSTM train example that shows accuracy, loss, while training.

## Translation
A bilinear LSTM (bi-LSTM) for `EN-FR` translation with a local hosted saved model. Full examples: load model, translate sentences.

## LSTM Text Generation
A bi-LSTM model that learns to write as Nietzsche or your favorite text. Work in progress, so far it loads the saved model only.

## Object Detection ::new::
An example that show how to load a Tensorflow SavedModel, without any conversion, using `tf.node.loadSavedModel`. It loads the model, predict the bounding boxes of detection objects and write to an image using a `canvas`.

### Can I use async/await without ECMA2017
Yes! Definitively you can. See `async2promise.js`. You have here defined two function the `__awaiter` and the `__generator`, that mimic the `async` and `await` using a `Promise`. Enjoy ECMA5.
