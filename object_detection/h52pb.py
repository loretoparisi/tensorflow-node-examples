import sys
from keras.models import load_model
import numpy as np

model = load_model(sys.argv[1])

from keras import backend as K
import tensorflow as tf

signature = tf.saved_model.signature_def_utils.predict_signature_def(   
    inputs={'image': model.input}, outputs={'scores': model.output})

builder = tf.saved_model.builder.SavedModelBuilder(sys.argv[2])
builder.add_meta_graph_and_variables(
    sess=K.get_session(),                                      
    tags=[tf.saved_model.tag_constants.SERVING],                                      
    signature_def_map={                                      
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:                           
            signature                       
    })                                      
builder.save()