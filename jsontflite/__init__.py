import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class jamestf:
    def __init__(self,modelpath):
        self.interpreter = tf.lite.Interpreter(model_path=modelpath)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def inference(self,frame,size):
        image_tensor = self.opencv2tensor(frame,size[0],size[1])
        self.interpreter.set_tensor(self.input_details[0]['index'], image_tensor)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def opencv2tensor(self,frame, input_height=224, input_width=224):
        float_caster = tf.cast(frame, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize(dims_expander, [input_height, input_width])
        sess = tf.Session()
        result = sess.run(resized)
        return result
