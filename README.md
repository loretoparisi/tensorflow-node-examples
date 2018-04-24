# tensorflow-node-examples
Tensorflow Node.js Examples

## Disclaimer
Tensorflow.js for Node.js (namely `tfjs-node`) is currently in development status, see [here](https://github.com/tensorflow/tfjs-node) for changes. The provided demo examples are written TypeScript. The aim of this repo is to provide simpler Node.js examples that run as-it-is, without any need of `tsc` (TypeScript compiler) toolchain when it is possible.

## getting started
To run the example first install dependencies and link locally the `tfjs-node` module
```
cd getting_started
yarn
yarn link-local tfjs-node
```

You can now run the example
```
node index.js
Tensor {
  isDisposed: false,
  size: 1,
  shape: [ 1, 1 ],
  dtype: 'float32',
  strides: [ 1 ],
  dataId: {},
  id: 12,
  rankType: '2' }
```
