# load model
from tensorflow.keras.models import load_model
model = load_model('model_cat_dog.h5')

import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('mage/dog.4030.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0]<=0.5:
    print("The image classified is cat")
else:
    print("The image classified is dog")