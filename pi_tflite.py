# load model
import tensorflow as tf
interpreter = tf.lite.Interpreter('model_cat_dog.tflite')
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input Shape: ",input_details[0]['shape'])
print("Input Type: ",input_details[0]['dtype'])
print("Output Shape: ",output_details[0]['shape'])
print("Output Type: ",output_details[0]['dtype'])

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('image/dog.4030.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)

print(test_image.dtype)
print(test_image.shape)

# Test model on random input data.
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

if output_data[0]<=0.5:
    print("The image classified is cat")
else:
    print("The image classified is dog")