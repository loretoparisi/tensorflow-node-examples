/**
 * Tensorflow.js Examples for Node.js
 * Script adatapted from 
 * https://github.com/tensorflow/tfjs-examples
 * https://groups.google.com/a/tensorflow.org/forum/#!forum/tfjs
 * @author Loreto Parisi (loretoparisi@gmail.com)
 * @author Simone Francia (francia.simone1@gmail.com)
 * @copyright 2018 Loreto Parisi (loretoparisi@gmail.com)
 */

// The error occurs because tfjs-node currently uses `fetch` to send HTTP requests, but `fetch` is not available in Node.js by default. 
global.fetch = require('node-fetch');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');


const model_path = 'file://./model/en-fr/model.json';
const model_metadata = __dirname + '/model/en-fr/metadata.json';

let Translator = function () {

    this.loadMetadata = () => {

        const translationMetadata = JSON.parse(fs.readFileSync(model_metadata));
        this.maxDecoderSeqLength = translationMetadata['max_decoder_seq_length'];
        this.maxEncoderSeqLength = translationMetadata['max_encoder_seq_length'];
        console.log('maxDecoderSeqLength = ' + this.maxDecoderSeqLength);
        console.log('maxEncoderSeqLength = ' + this.maxEncoderSeqLength);
        this.inputTokenIndex = translationMetadata['input_token_index'];
        this.targetTokenIndex = translationMetadata['target_token_index'];
        this.reverseTargetCharIndex =
            Object.keys(this.targetTokenIndex)
                .reduce(
                    (obj, key) => (obj[this.targetTokenIndex[key]] = key, obj), {});

    }

    this.loadModel = () => new Promise((resolve, reject) => {
        let self = this;
        tf.loadLayersModel(model_path)
            .then(model => {
                model.summary();
                self.loadMetadata();
                resolve(model);
            })
            .catch(error => {
                console.error(error)
                reject(error)
            })
    })

    this.prepareEncoderModel = (model) => {
        this.numEncoderTokens = model.input[0].shape[2];
        console.log('numEncoderTokens = ' + this.numEncoderTokens);

        const encoderInputs = model.input[0];
        const stateH = model.layers[2].output[1];
        const stateC = model.layers[2].output[2];
        const encoderStates = [stateH, stateC];

        this.encoderModel =
            tf.model({ inputs: encoderInputs, outputs: encoderStates });
    }

    this.prepareDecoderModel = (model) => {
        this.numDecoderTokens = model.input[1].shape[2];
        console.log('numDecoderTokens = ' + this.numDecoderTokens);

        const stateH = model.layers[2].output[1];
        const latentDim = stateH.shape[stateH.shape.length - 1];
        console.log('latentDim = ' + latentDim);
        const decoderStateInputH =
            tf.input({ shape: [latentDim], name: 'decoder_state_input_h' });
        const decoderStateInputC =
            tf.input({ shape: [latentDim], name: 'decoder_state_input_c' });
        const decoderStateInputs = [decoderStateInputH, decoderStateInputC];

        const decoderLSTM = model.layers[3];
        const decoderInputs = decoderLSTM.input[0];
        const applyOutputs =
            decoderLSTM.apply(decoderInputs, { initialState: decoderStateInputs });
        let decoderOutputs = applyOutputs[0];
        const decoderStateH = applyOutputs[1];
        const decoderStateC = applyOutputs[2];
        const decoderStates = [decoderStateH, decoderStateC];

        const decoderDense = model.layers[4];
        decoderOutputs = decoderDense.apply(decoderOutputs);
        this.decoderModel = tf.model({
            inputs: [decoderInputs].concat(decoderStateInputs),
            outputs: [decoderOutputs].concat(decoderStates)
        });
    }

    /**
     * Encode a string (e.g., a sentence) as a Tensor3D that can be fed directly
     * into the TensorFlow.js model.
     */
    this.encodeString = (str) => {
        const strLen = str.length;
        const encoded =
            tf.buffer([1, this.maxEncoderSeqLength, this.numEncoderTokens]);
        for (let i = 0; i < strLen; ++i) {
            if (i >= this.maxEncoderSeqLength) {
                console.error(
                    'Input sentence exceeds maximum encoder sequence length: ' +
                    this.maxEncoderSeqLength);
            }

            const tokenIndex = this.inputTokenIndex[str[i]];
            if (tokenIndex == null) {
                console.error(
                    'Character not found in input token index: "' + tokenIndex + '"');
            }
            encoded.set(1, 0, i, tokenIndex);
        }
        return encoded.toTensor();
    }

    this.decodeSequence = (inputSeq) => {
        // Encode the inputs state vectors.
        let statesValue = this.encoderModel.predict(inputSeq);

        // Generate empty target sequence of length 1.
        let targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
        // Populate the first character of the target sequence with the start
        // character.
        targetSeq.set(1, 0, 0, this.targetTokenIndex['\t']);

        // Sample loop for a batch of sequences.
        // (to simplify, here we assume that a batch of size 1).
        let stopCondition = false;
        let decodedSentence = '';
        while (!stopCondition) {
            const predictOutputs =
                this.decoderModel.predict([targetSeq.toTensor()].concat(statesValue));
            const outputTokens = predictOutputs[0];
            const h = predictOutputs[1];
            const c = predictOutputs[2];

            // Sample a token.
            // We know that outputTokens.shape is [1, 1, n], so no need for slicing.
            const logits = outputTokens.reshape([outputTokens.shape[2]]);
            const sampledTokenIndex = logits.argMax().dataSync()[0];
            const sampledChar = this.reverseTargetCharIndex[sampledTokenIndex];
            decodedSentence += sampledChar;

            // Exit condition: either hit max length or find stop character.
            if (sampledChar === '\n' ||
                decodedSentence.length > this.maxDecoderSeqLength) {
                stopCondition = true;
            }

            // Update the target sequence (of length 1).
            targetSeq = tf.buffer([1, 1, this.numDecoderTokens]);
            targetSeq.set(1, 0, 0, sampledTokenIndex);

            // Update states.
            statesValue = [h, c];
        }

        return decodedSentence;
    }

    /** Translate the given English sentence into French. */
    this.translate = (inputSentence) => {
        const inputSeq = this.encodeString(inputSentence);
        const decodedSentence = this.decodeSequence(inputSeq);
        return decodedSentence;
    }

};

var translator = new Translator();
translator.
    loadModel()
    .then(model => {

        translator.prepareEncoderModel(model);
        translator.prepareDecoderModel(model);

        console.log(translator.translate("they're"));
        console.log(translator.translate("they're cool"));
        console.log(translator.translate("they're safe"));

    })



