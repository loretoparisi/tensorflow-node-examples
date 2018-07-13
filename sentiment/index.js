/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018 Loreto Parisi (loretoparisi@gmail.com)
 */
require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

var fs = require('fs');
var performance = require('perf_hooks').performance;

const model_path = 'file://' + __dirname + '/model/model.json';
const model_metadata = __dirname + '/model/metadata.json';

var text = 'this is a bad day';

tf.loadModel(model_path)
    .then(model => {

        let sentimentMetadata = JSON.parse(fs.readFileSync(model_metadata));
        //console.log(sentimentMetadata);

        let indexFrom = sentimentMetadata['index_from'];
        let maxLen = sentimentMetadata['max_len'];
        let wordIndex = sentimentMetadata['word_index'];

        console.log('indexFrom = ' + indexFrom);
        console.log('maxLen = ' + maxLen);

        console.log('model_type', sentimentMetadata['model_type']);
        console.log('vocabulary_size', sentimentMetadata['vocabulary_size']);
        console.log('max_len', sentimentMetadata['max_len']);

        const inputText =
            text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(/\s+/g); // tokenized

        console.log(inputText);

        // Look up word indices.
        const inputBuffer = tf.buffer([1, maxLen], 'float32');
        for (let i = 0; i < inputText.length; ++i) {
            const word = inputText[i];
            if (typeof wordIndex[word] == 'undefined') { // TODO(cais): Deal with OOV words.
                console.log(word, wordIndex[word]);
            }
            inputBuffer.set(wordIndex[word] + indexFrom, 0, i);
        }
        const input = inputBuffer.toTensor();

        console.log(text, "\n", input);

        const beginMs = performance.now();
        const predictOut = model.predict(input);
        const score = predictOut.dataSync()[0];
        predictOut.dispose();
        const endMs = performance.now();

        console.log({ score: score, elapsed: (endMs - beginMs) });



    })
    .catch(error => {
        console.error(error)
    })