const tfjsnode = require('@tensorflow/tfjs-node');
const tfconv = require("@tensorflow/tfjs-converter");

var loadGraphModel = function (url) {
  return new Promise(function (resolve, reject) {
    tfconv.loadGraphModel(url,
      { fromTFHub: true })
      .then(res => {
        console.log("loadGraphModel OK");
        resolve(res);
      })
      .catch(err => reject(err));
  });
}
var loadSavedModel = function (path) {
  return new Promise(function (resolve, reject) {
    tfjsnode.node.loadSavedModel(path)
      .then(res => {
        console.log("loadSavedModel OK");
        resolve(res);
      })
      .catch(err => reject(err));
  });
}
loadGraphModel('https://tfhub.dev/tensorflow/tfjs-model/toxicity/1/default/1')
  .catch(err => console.error("loadGraphModel", err));
loadSavedModel('./toxicity_model/saved')
  .catch(err => console.error("loadSavedModel", err));
