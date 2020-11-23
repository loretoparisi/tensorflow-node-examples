const tf = require('@tensorflow/tfjs-node')
const use = require("@tensorflow-models/universal-sentence-encoder");

async function run() {
    const model = await tf.node.loadSavedModel('./toxicity_model/saved');
    const tokenizer = await use.load();
    const sentences = ['you suck', 'hello how are you?'];

    var encodings = await tokenizer.embed(sentences);
    console.log(encodings);
    
    /*var indicesArr = encodings.map(function (arr, i) { return arr.map(function (d, index) { return [i, index]; }); });
    var flattenedIndicesArr = [];
    for (i = 0; i < indicesArr.length; i++) {
        flattenedIndicesArr =
            flattenedIndicesArr.concat(indicesArr[i]);
    }
    var indices = tf.tensor2d(flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
    var values = tf.tensor1d(tf.util.flatten(encodings), 'int32');
    */

    // both indexArray and valueArray are obtained from two preprocessed test phrases that I used to verify
    // model outputs
    const indexArray = [
      [0, 1], [0,2 ], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8],
      [1, 0], [1, 1], [1, 2], [1, 3]
    ]
    const valueArray = [215, 13, 53, 4461, 2951, 519, 1129, 7, 78, 16, 123, 20, 6]
    const indices = tf.tensor2d(indexArray, [indexArray.length, 2], 'int32')
    const values = tf.tensor1d(valueArray, 'int32')
    
    const modelInputs = {
        Placeholder_1: indices,
        Placeholder: values
    }
    const labels = model.predict(modelInputs)
    indices.dispose()
    values.dispose()
    outputs = []
    for (name in labels) {
        const prediction = labels[name].dataSync()
        const results = []
        for (let input = 0; input < 2; ++input) {
            const probs = prediction.slice(input * 2, input * 2 + 2)
            let match = null
            if (Math.max(probs[0], probs[1]) > 0.9) {
                match = probs[0] > probs[1]
            }
            p = probs.toString() // just to print out the numbers
            results.push({ p, match })
        }
        outputs.push({ label: name.split('/')[0], results })
    }
    for (x of outputs) {
        console.log(x)
    }

    const modelInfo = await tf.node.getMetaGraphsFromSavedModel('./toxicity_model/saved');
    console.dir(modelInfo[0].signatureDefs.serving_default.outputs, { depth: null, maxArrayLength: null });

}

run()