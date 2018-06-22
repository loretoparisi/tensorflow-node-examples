/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018 Loreto Parisi (loretoparisi@gmail.com)
 */
require('@tensorflow/tfjs-node');
var tf = require('@tensorflow/tfjs');
tf.setBackend('tensorflow');
function myFirstTfjs() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });
  // Generate some synthetic data for training. (y = 2x - 1)                                                                                                            
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
  // Train the model using the data.                                                                                                                                    
  model.fit(xs, ys, { epochs: 500 })
      .then(res => {
        p10 = model.predict(tf.tensor2d([5.5], [1, 1]))
        pExp10 = tf.exp(p10);
        console.log("2*5.5 - 1 SHOULD BE CLOSE TO 10");
        console.log(p10.dataSync());
        console.log("exp(10) SHOULD BE CLOSE TO 22026");
        console.log(pExp10.dataSync());
      })
      .catch(error => {
        console.error(error);
      })
}
myFirstTfjs();