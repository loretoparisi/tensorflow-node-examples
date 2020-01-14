/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @copyright 2018-2019 Loreto Parisi (loretoparisi@gmail.com)
 */


const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const TextData = require('./data');

var generatedTextInput = '';

/**
 * A function to call each time a character is obtained during text generation.
 *
 * @param {string} char The just-generated character.
 */
function onTextGenerationChar(char) {
    generatedTextInput += char;
    const charCount = generatedTextInput.length;
    const status = `Generating text: ${charCount} ${generatedTextInput} complete...`;
    console.log(status)
}

/**
* Draw a sample based on probabilities.
*
* @param {tf.Tensor} probs Predicted probability scores, as a 1D `tf.Tensor` of
*   shape `[charSetSize]`.
* @param {tf.Tensor} temperature Temperature (i.e., a measure of randomness
*   or diversity) to use during sampling. Number be a number > 0, as a Scalar
*   `tf.Tensor`.
* @returns {number} The 0-based index for the randomly-drawn sample, in the
*   range of `[0, charSetSize - 1]`.
*/
function sample(probs, temperature) {
    return tf.tidy(() => {
        const logits = tf.div(tf.log(probs), Math.max(temperature, 1e-6));
        const isNormalized = false;
        // `logits` is for a multinomial distribution, scaled by the temperature.
        // We randomly draw a sample from the distribution.
        return tf.multinomial(logits, 1, null, isNormalized).dataSync()[0];
    });
}

/**
 * Generate text using a next-char-prediction model.
 *
 * @param {tf.Model} model The model object to be used for the text generation,
 *   assumed to have input shape `[null, sampleLen, charSetSize]` and output
 *   shape `[null, charSetSize]`.
 * @param {number[]} sentenceIndices The character indices in the seed sentence.
 * @param {number} length Length of the sentence to generate.
 * @param {number} temperature Temperature value. Must be a number >= 0 and
 *   <= 1.
 * @param {(char: string) => Promise<void>} onTextGenerationChar An optinoal
 *   callback to be invoked each time a character is generated.
 * @returns {string} The generated sentence.
 */
function generateText(
    model, textData, sentenceIndices, length, temperature,
    onTextGenerationChar) {
    const sampleLen = model.inputs[0].shape[1];
    const charSetSize = model.inputs[0].shape[2];

    // Avoid overwriting the original input.
    sentenceIndices = sentenceIndices.slice();

    let generated = '';
    while (generated.length < length) {
        // Encode the current input sequence as a one-hot Tensor.
        const inputBuffer =
            new tf.TensorBuffer([1, sampleLen, charSetSize]);

        // Make the one-hot encoding of the seeding sentence.
        for (let i = 0; i < sampleLen; ++i) {
            inputBuffer.set(1, 0, i, sentenceIndices[i]);
        }
        const input = inputBuffer.toTensor();

        // Call model.predict() to get the probability values of the next
        // character.
        const output = model.predict(input);

        // Sample randomly based on the probability values.
        const winnerIndex = sample(tf.squeeze(output), temperature);
        const winnerChar = textData.getFromCharSet(winnerIndex);
        if (onTextGenerationChar != null) {
            onTextGenerationChar(winnerChar);
        }

        generated += winnerChar;
        sentenceIndices = sentenceIndices.slice(1);
        sentenceIndices.push(winnerIndex);

        // Memory cleanups.
        input.dispose();
        output.dispose();
    }
    return generated;
}

const FAKE_TEXT = `Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse tempor aliquet justo non varius. Curabitur eget convallis velit. Vivamus malesuada, tortor ut finibus posuere, libero lacus eleifend felis, sit amet tempus dolor magna id nibh. Praesent non turpis libero. Praesent luctus, neque vitae suscipit suscipit, arcu neque aliquam justo, eget gravida diam augue nec lorem. Etiam scelerisque vel nibh sit amet maximus. Praesent et dui quis elit bibendum elementum a eget velit. Mauris porta lorem ac porttitor congue. Vestibulum lobortis ultrices velit, vitae condimentum elit ultrices a. Vivamus rutrum ultrices eros ac finibus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Morbi a purus a nibh eleifend convallis. Praesent non turpis volutpat, imperdiet lacus in, cursus tellus. Etiam elit velit, ornare sit amet nulla vel, aliquam iaculis mauris.
Phasellus sed sem ut justo sollicitudin cursus at sed neque. Proin tempor finibus nisl, nec aliquam leo porta at. Nullam vel mauris et neque pellentesque laoreet sit amet eu risus. Sed sed ante sed enim hendrerit commodo. Etiam blandit aliquet molestie. Nullam dictum imperdiet enim, quis scelerisque nunc ultricies sit amet. Praesent dictum dictum lobortis. Sed ut ipsum at orci commodo congue.
Aenean pharetra mollis erat, id convallis ante elementum at. Cras semper turpis nec lorem tempus ultrices. Sed eget purus vel est blandit dictum. Praesent auctor, sapien non consequat pellentesque, risus orci sagittis leo, at cursus nibh nisi vel quam. Morbi et orci id quam dictum efficitur ac iaculis nisl. Donec at nunc et nibh accumsan malesuada eu in odio. Donec quis elementum turpis. Vestibulum pretium rhoncus orci, nec gravida nisl hendrerit pellentesque. Cras imperdiet odio a quam mollis, in aliquet neque efficitur. Praesent at tincidunt ipsum. Maecenas neque risus, pretium ut orci sit amet, dignissim auctor dui. Sed finibus nunc elit, rhoncus ornare dui pharetra vitae. Sed ut iaculis ex. Quisque quis molestie ligula. Vivamus egestas rhoncus mollis.
Pellentesque volutpat ipsum vitae ex interdum, eu rhoncus dolor fringilla. Suspendisse potenti. Maecenas in sem leo. Curabitur vestibulum porta vulputate. Nunc quis consectetur enim. Aliquam congue, augue in commodo porttitor, sem tellus posuere augue, ut aliquam sapien massa in est. Duis convallis pellentesque vehicula. Mauris ipsum urna, congue consequat posuere sed, euismod nec mauris. Praesent sollicitudin scelerisque scelerisque. Ut commodo nisl vitae nunc feugiat auctor. Praesent imperdiet magna facilisis nunc vulputate, vel suscipit leo consequat. Duis fermentum rutrum ipsum a laoreet. Nunc dictum libero in quam pellentesque, sit amet tempus tellus suscipit. Curabitur pharetra erat bibendum malesuada rhoncus.
Donec laoreet leo ligula, ut condimentum mi placerat ut. Sed pretium sollicitudin nisl quis tincidunt. Proin id nisl ornare, interdum lorem quis, posuere lacus. Cras cursus mollis scelerisque. Mauris mattis mi sed orci feugiat, et blandit velit tincidunt. Donec ultrices leo vel tellus tincidunt, id vehicula mi commodo. Nulla egestas mollis massa. Etiam blandit nisl eu risus luctus viverra. Mauris eget mi sem.
`;

function createTextDataForTest(sampleLen, sampleStep = 1) {
    return new TextData('LoremIpsum', FAKE_TEXT, sampleLen, sampleStep);
}

function readTextData(localTextDataPath, sampleLen, sampleStep = 1) {
    const text = fs.readFileSync(localTextDataPath, { encoding: 'utf-8' });
    const textData =
        new TextData('text-data', text, sampleLen, sampleStep);
    return textData;
}

let model_path = 'file://./model/nietzsche.json';
tf.loadLayersModel(model_path)
    .then(model => {

        model.summary();

        const sampleLen = 1000;
        const generateLength = 250
        const temperature = 0.6

        const textData = readTextData('./data/nietzsche.txt', sampleLen);
        
        let seedSentence;
        let seedSentenceIndices;
        [seedSentence, seedSentenceIndices] = textData.getRandomSlice();

        let generated = generateText(model, textData, seedSentenceIndices, generateLength, temperature,
            onTextGenerationChar);

        console.log(generated);

    })
    .catch(error => {
        console.error(error)
    })



