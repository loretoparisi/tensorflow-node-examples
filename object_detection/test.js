const tf = require('@tensorflow/tfjs-node');

const model_path = '/root/spleeter_pretrained_models/saved_model';
tf.node.loadSavedModel(model_path, ['serve'], 'serving_default')
.then(res => {
    console.log(res);

    const waveformTensor;

    /**
     * prediction = predictor({
            'waveform': waveform,
            'audio_id': ''})
     */
    let outputTensor = res.predict({
        audio_id: '',
        waveform: waveformTensor,
    });
    console.log(outputTensor);

    return tf.node.getMetaGraphsFromSavedModel(model_path);
})
.then(modelInfo => {
    console.log(modelInfo);
})
.catch(error => {
    console.error(error)
})