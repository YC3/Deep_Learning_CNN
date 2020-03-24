import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras import models
import operator
import sys, getopt
import cv2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
Parameters:
-o, --original: bool, whether show the original image or not.
-c, --cutoff: float, the cutoff (between 0 to 1) used to choose significant genes.

Example:
python3 CNN_gene_picker.py -o False -c 0.7

''' 



def feature_filter(data):
    
    # features needed to form an sqaure image
    dim = int(np.sqrt(data.shape[1]))
    top = dim**2
    
    # filtering out features with low variances across examples
    std_v = data.std(axis = 0)
    selected = std_v.sort_values(ascending = False)[0:top]
    output = data.loc[:, selected.index]
    
    ncol = output.shape[1]
    rand_id = np.random.choice(range(0, ncol), ncol, replace = False)
    output_rand = output.iloc[:, rand_id]
    
    return output_rand, dim


def Df2Image(data, dim):

    image = np.empty((data.shape[0], dim, dim, 1))
    
    for i in range(data.shape[0]):
        image[i,:,:,:] = np.array(data.iloc[i, :]).reshape(dim, dim, 1)
    
    return image


def scaling01(data):

    dmin = data.min()
    dmax = data.max()
    return (data - dmin)/(dmax - dmin)
    
    
def image_scaling(tensor4D):
    
    output = np.empty(tensor4D.shape)
    for i in range(tensor4D.shape[0]):
        example = tensor4D[i]
        scaled_example = scaling01(example)
        output[i] = scaled_example.reshape(tensor4D.shape[1], tensor4D.shape[2], 1)    
    
    return output


def gGradCAM(conv_model, data, img_num, class_num, conv_layer, if_save_img):

    # selected image
    img = data[img_num] 
    img_tensor = np.array([img])

    # gradients of selected conv layer
    grad_model = tf.keras.models.Model([conv_model.inputs], [conv_model.layers[conv_layer].output, conv_model.output])

    with tf.GradientTape() as tape:
        featureMaps, predictions = grad_model(img_tensor)
        loss = predictions[:, class_num] # class number

    grads = tape.gradient(loss, featureMaps)[0]  # gradients for all channels
    featr = featureMaps[0]                       # activations for all channels

    # ReLU
    grads = tf.cast(grads > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    # weighted average of all feature maps (\sum_k{\alpha^c_k \cdot A^k}) 
    alpha = tf.reduce_mean(grads, axis=(0, 1))  # Gradients average (\alpha^c_k)

    out_img1 = np.ones(featr.shape[0: 2], dtype = np.float32)
    for k, a in enumerate(alpha):
        out_img1 += a * featr[:, :, k]

    # processing output img
    out_img2 = cv2.resize(out_img1.numpy(), (dim, dim))  # bi-linear interpolation
    output = (out_img2 - out_img2.min()) / (out_img2.max() - out_img2.min())  # scaling
    
    # plotting image
    plt.imshow(output)
    plt.show()
        
    return output



def GetGenes(data, cutoff): 
    
    sigPixels = np.zeros(data.shape)
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            if data[i, j] > cutoff:
                sigPixels[i, j] = 1
    sigPixels = np.array(sigPixels).reshape(1, len(sigPixels)**2)[0]
    
    allGenes = data_filtered.columns.values
    sigGenes = []
    for i in range(0, len(sigPixels)):
        if sigPixels[i] == 1:
            sigGenes = sigGenes + [allGenes[i]]

    return sigGenes



if __name__ == '__main__':

    # get user input
    opts,args = getopt.getopt(sys.argv[1:], '-o:-c:', ['original=', 'cutoff='])
    for opt_name, opt_value in opts:
        if opt_name in ('-o', '--original'):
            if_plot_original = eval(opt_value)
        if opt_name in ('-c', '--cutoff'):
            cutoff = float(opt_value)

    # read in data
    data = pd.read_csv('tumor_data.csv', index_col = 0)

    # read in CNN model
    cnn = tf.keras.models.load_model('model_cnn3_val.h5')
    
    # embed dataFrame to image
    data_filtered, dim = feature_filter(data)
    data_image = Df2Image(data_filtered, dim)
    data_scaled_image = image_scaling(data_image)
    
    # plot the original image
    if if_plot_original:
        img_tensor = data_scaled_image[1]
        plt.imshow(img_tensor.reshape(dim, dim))
        plt.show()

    # identifying significant pixels using guided Grad-CAM
    cam_out = gGradCAM(conv_model = cnn, data = data_scaled_image, img_num = 0, class_num = 1, conv_layer = 0, if_save_img = False)

    # extract significant genes for selected class
    topGenes = GetGenes(cam_out, cutoff = cutoff)
    print('Selected', len(topGenes), 'genes:')
    print(topGenes)
