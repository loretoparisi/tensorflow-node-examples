/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018 Loreto Parisi (loretoparisi@gmail.com)
 */
const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

var fs = require('fs');
var performance = require('perf_hooks').performance;

//
// Script adapted from: https://codepen.io/caisq/pen/vrxOvy
//
// TensorFlow.js example: Trains LSTM model to perform the following sequence task:
//
// Given a sequence of 0s and 1s of fixed length (10), output a single binary number (0 or 1).
//
// The training data has the following pattern:
//
// The output (i.e., label) is 1 if there are four or more consecutive and identical
// items (either 0s or 1s) in the input sequence. Otherwise, the output is 0. For example:
//   Sequence [0, 1, 0, 1, 0, 1, 0, 0, 1, 0] --> Label: 0.
//   Sequence [0, 1, 1, 1, 1, 0, 1, 0, 0, 1] --> Label: 1.
//   Sequence [0, 0, 0, 0, 0, 0, 1, 0, 0, 1] --> Label: 1.

const sequenceLength = 10;
const stretchLengthThreshold = 4;

// Generates sequences consisting of 0 and 1 and the associated 0-1 labels.
//  
// The sequences and labels follow the pattern described above.
//
// Args:
//   len: Length of the sequence.
// 
// Returns:
//   1. An Array of randomly-generated 0s and 1s.
//   2. The associated output (label): a 0 or a 1.
function generateSequenceAndLabel(len) {
    const sequence = [];
    let currentItem = -1;
    let stretchLength = 0;
    let label = 0;
    for (let i = 0; i < len; ++i) {
        const item = Math.random() > 0.5 ? 1 : 0;
        sequence.push(item);
        if (currentItem === item) {
            stretchLength++;
        } else {
            currentItem = item;
            stretchLength = 1;
        }
        if (stretchLength >= stretchLengthThreshold) {
            label = 1;
        }
    }
    return [sequence, label];
}

// Generates a dataset consisting of sequences and their corresponding labels.
//
// Args:
//   numExamples: Number of examples to generate.
//   sequenceLength: Length of each individual sequence.
//
// Returns:
//   1. Sequence Tensor: a Tensor of shape [numExamples, sequenceLength, 2].
//      The first dimension is the batch examples.
//      The second dimension is the time axis (sequence items).
//      The third dimension is the one-hot encoding of the 0/1 items.
//   2. Label Tensor: a Tensor of shape [numExamples, 1].
//      Each element of this Tensor is 0 or 1. 
function generateDataset(numExamples, sequenceLength) {
    const sequencesBuffer = tf.buffer([numExamples, sequenceLength, 2]);
    const labelsBuffer = tf.buffer([numExamples, 1]);
    for (let i = 0; i < numExamples; ++i) {
        const [sequence, label] = generateSequenceAndLabel(sequenceLength);
        for (let j = 0; j < sequenceLength; ++j) {
            sequencesBuffer.set(1, i, j, sequence[j]);
        }
        labelsBuffer.set(label, i, 0);
    }
    return [sequencesBuffer.toTensor(), labelsBuffer.toTensor()];
}

tf.nextFrame = function () {
    return new Promise((resolve, reject) => {
        process.nextTick(function () {
            return resolve(true);
        });
    });
}

// Train a model to predict the label based on the sequence.
function train() {
    return new Promise((resolve, reject) => {
        // Define the topology of the model.
        const model = tf.sequential();
        model.add(tf.layers.lstm({ units: 8, inputShape: [sequenceLength, 2] }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        // Compile model to prepare for training.
        const learningRate = 4e-3;
        const optimizer = tf.train.rmsprop(learningRate);
        model.compile({
            loss: 'binaryCrossentropy',
            optimizer: optimizer,
            metrics: ['acc']
        });

        // Generate a number of examples for training.
        const numTrainExamples = 500;
        console.log('Generating training data...');
        const [trainSequences, trainLabels] = generateDataset(numTrainExamples, 10);

        let status = {
            train_epoch: 0,
            train_loss: 0,
            train_acc: 0,
            val_loss: 0,
            val_acc: 0
        };
        console.log('Training model...');
        model.fit(
            trainSequences, trainLabels, {
                epochs: 5,
                validationSplit: 0.15,
                callbacks: {
                    onBatchEnd: (batch, logs) => {
                        console.log(batch,logs);
                        tf.nextFrame().then(res => {
                            return;
                        })
                    },
                    onEpochEnd: (epoch, logs) => {
                        // Update the UI to display the current loss and accuracy values.
                        status.train_epoch = epoch + 1;
                        status.train_loss = logs.loss;
                        status.train_acc = logs.acc;
                        status.val_loss = logs.val_loss;
                        status.val_acc = logs.val_acc;
                        console.log(status);
                        tf.nextFrame().then(res => {
                            return;
                        })
                    },
                }
            }).then(fitOutput => {
                // Memory clean up: Dispose the training data.
                trainSequences.dispose();
                trainLabels.dispose();
                return resolve(fitOutput);
            })
            .catch(error => {
                return reject(error);
            })
    })
}

train()
    .then(res => {
        console.log(res);
    })
    .catch(error => {
        console.error(error);
    })