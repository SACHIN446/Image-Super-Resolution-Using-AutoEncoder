import os
import re
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, MaxPooling2D, Dropout, UpSampling2D,add
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


#----------PREPARE DATASET----------
def GET_DATASET(number_of_batches = 27):
    n_batch = number_of_batches
    bn = 0
    low_batch_array = []
    high_batch_array = []
    img_high_ds = []
    img_low_ds = []
    for r, n, f in os.walk('/home/sachin/MY_FOLDER/PYCHARM/Tensorflow/Datasets/Image Super Resolution/bmw10'):
        for file in f:
            if re.search('.(jpg|jpeg|png|bmf|tiff)\Z', file, re.I):
                img_path = os.path.join(r, file)
                img = plt.imread(img_path)/255
                if len(img.shape) > 2:
                    img_resize = resize(img, (256, 256))
                    high_batch_array.append(img_resize)
                    low_batch_array.append(rescale(rescale(img_resize,0.5,multichannel=True), 2.0,multichannel=True))
                    bn += 1
                    if bn == n_batch:
                        img_high_ds = np.array(high_batch_array)
                        img_low_ds = np.array(low_batch_array)
                        bn = 0
                        high_batch_array = []
                        low_batch_array = []

            else:
                print("Invalid Image Extension Found:", file)
    return img_low_ds, img_high_ds

#----------ENCODER PART----------
def ENCODER():
    input = Input(shape=(256,256,3))
    layer1 = Conv2D(64, (3, 3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(input)
    layer2 = Conv2D(64, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer1)
    layer3 = MaxPooling2D(padding='same')(layer2)
    layer4 = Dropout(0.3)(layer3)
    layer5 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer4)
    layer6 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer5)
    layer7 = MaxPooling2D(padding='same')(layer6)
    output = Conv2D(256, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer7)
    encoder = Model(input, output)
    return encoder

def AUTO_ENCODER():
    ip = Input(shape=(256, 256, 3))
    layer1 = Conv2D(64, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(ip)
    layer2 = Conv2D(64, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer1)
    layer3 = MaxPooling2D(padding='same')(layer2)
    layer4 = Dropout(0.3)(layer3)
    layer5 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer4)
    layer6 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer5)
    layer7 = MaxPooling2D(padding='same')(layer6)
    layer8 = Conv2D(256, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer7)
    layer9 = UpSampling2D()(layer8)
    layer10 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer9)
    layer11 = Conv2D(128, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer10)
    layer12 = add([layer6, layer11])
    layer13 = UpSampling2D()(layer12)
    layer14 = Conv2D(64, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer13)
    layer15 = Conv2D(64, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer14)
    layer16 = add([layer15, layer2])
    op = Conv2D(3, (3,3), padding='same', activation='relu', activity_regularizer=regularizers.l1(10e-10))(layer16)
    auto_enc = Model(ip, op)
    return auto_enc

low_ds, high_ds = GET_DATASET()

auto_encoder = AUTO_ENCODER()
#auto_encoder.summary()
auto_encoder.compile(optimizer='adadelta', loss='mean_squared_error')
#auto_encoder.load_weights('/home/sachin/MY_FOLDER/PYCHARM/Tensorflow/Datasets/Image Super Resolution/sr.img_net.mse.final_model5.no_patch.weights.best.hdf5')
auto_encoder.fit(low_ds,high_ds, epochs=20)

res = auto_encoder.predict(low_ds)
print("result=",res[0].shape, np.max(res[0]), np.min(res[0]))
print(low_ds[0].shape, np.max(low_ds[0]), np.min(low_ds[0]))

i = np.random.randint(0,27)
plt.figure(figsize=(20,20))
plt.subplot(1,3,1)
plt.imshow(low_ds[i])
plt.title("low res")

plt.subplot(1,3,2)
plt.imshow(res[i])
plt.title("Predicted")

plt.subplot(1,3,3)
plt.imshow(high_ds[i])
plt.title("High Res")
plt.show()







