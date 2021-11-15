# Image-Super-Resolution-Using-AutoEncoder
This Project deals with Generating high quality image from the existing one wihtout changing image resolution

## Introduction
Traditional method for image enchancement were interpolation techinques which might use Linear, Bicubic, Nearest Neighbour methods.
When Deep Neural Network is used to learn the pattern and reconstruct the same image, the results were better than traditional method still there are some error because of less training short dataset.

Although the advantage of this project is, it is not going to change image resolution (or increase) but give us better result than original one.

## Method
1. Dividing the high quality image in two parts, the first one is high quality image and second dataset is low quality image with same resolution pixels value.
2. Low quality image dataset is prepared using multiple scaling of image file which lead to add noise in image and this low quality image dataset will act as input to our model.
3. Once the dataset is prepeared and processed, we build our Auto Encoder Model, i made it using the model structure from one of the couse available on COURSERA plateform.
4. Now train the model or alternatively one can used the already saved model weights for better performance.

## Additional Details
One can use above python code to see the results.
