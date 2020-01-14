# object detection
Object detection example using [SavedModel](https://blog.tensorflow.org/2020/01/run-tensorflow-savedmodel-in-nodejs-directly-without-conversion.html)

## How to run
```
cd tensorflow-node-examples
npm install
cd object_detection
node index
```

You will get boxes predictions from test image:


```javascript
{
  boxes: [
    [
      0.1237560361623764,
      0.11185837537050247,
      0.47802025079727173,
      0.36011046171188354
    ],
    [ 0.40676987171173096, 0.23213016986846924, 0.8086241483688354, 1 ]
  ],
  names: [ 'dog', 'dog' ],
  inferenceTime: 9816.83417892456
}
```

and then loaded into `canvas` and saved to a new image `image_test.jpeg`

<p align="center">
    <img src="image_test.jpeg?raw=true" width="768"> </br>
</p>

## Disclaimer
Based on Tensorflow.js example https://github.com/tensorflow/tfjs-examples/tree/master/firebase-object-detection-node

## Convert to SavedModel
To convert a Keras `h5` model to a Tensorflow `pb` SavedModel just run the python script `h52pb.py` with: `python3 h52pb.py keras_model.h5 saved_model_folder`.

This will create the folder `saved_model_folder` with the following structure:

```
└── saved_model
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```
