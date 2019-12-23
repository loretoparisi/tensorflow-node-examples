/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018 Loreto Parisi (loretoparisi@gmail.com)
 */

// The error occurs because tfjs-node currently uses `fetch` to send HTTP requests, but `fetch` is not available in Node.js by default. 
global.fetch = require('node-fetch');
const tf = require('@tensorflow/tfjs-node');
let model_path = 'file://./model/nietzsche.json';
tf.loadLayersModel(model_path)
    .then(model => {
    
        model.summary();
})
.catch(error => {
    console.error(error)
})



