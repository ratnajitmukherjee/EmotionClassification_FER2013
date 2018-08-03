"""
 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2018, Ratnajit Mukherjee.
 " All rights reserved.
 "
 " Redistribution and use in source and binary forms, with or without
 " modification, are permitted provided that the following conditions are met:
 "
 " 1. Redistributions of source code must retain the above copyright notice,
 "    this list of conditions and the following disclaimer.
 "
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 "
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software
 "    without specific prior written permission.
 "
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: A VGG like network to extract facial expressions from the FER 2013 dataset and learn
                6 primary emotions (NOTE: we are merging 'anger' and 'disgust' into a single dataset
                due to lack of examples

                ====================================================================================
                Network Description:
                1) 8 Convolution layers (grouped as 2 x 4)
                2) 4 Maxpool layers
                3) 2 Densely connected layers
                4) 1 output layer with 6 classes
                ====================================================================================
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: July 2018
"""
# various imports to build the Neural Net
from keras.layers import Conv2D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras.models import Sequential
from keras import backend as K
import argparse
import numpy as np


def Emonet(num_classes):
    # use sequential model to build a VGG like network
    emonet = Sequential()

    """
    Convolution and Maxpool layers: Block 1
    """
    # Conv Layer 1:48x48x32
    emonet.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='linear', input_shape=(48, 48, 1)))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 2:48x48x32
    emonet.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 1
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.3))

    """
    Convolution and Maxpool layers: Block 2
    """
    # Conv Layer 3:24x24x64
    emonet.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 4:24x24x64
    emonet.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 2
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.3))

    """
    Convolution and Maxpool layers: Block 3
    """
    # Conv Layer 5:12x12x128
    emonet.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 6:12x12x128
    emonet.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 3
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.25))

    """
    Convolution and Maxpool layers: Block 4
    """
    # Conv Layer 7:6x6x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 8:6x6x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 4
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.25))

    # Flatten
    emonet.add(Flatten())

    # Dense layer 1:
    emonet.add(Dense(256, activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))
    emonet.add(Dropout(0.5))

    # Dense layer 2:
    emonet.add(Dense(256, activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))
    emonet.add(Dropout(0.5))

    # Output layer
    emonet.add(Dense(num_classes, activation='softmax'))

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(emonet.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(emonet.non_trainable_weights)]))

    # network summary
    print('\n\n---<summary>---')
    print('\n Layers: \n\tConvolution2D: {0}\n\tMaxPooling2D: {1}\n\tFully Connected Layers: {2}'.format(8, 4, 2))
    print('\n Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('\n Trainable params: {:,}'.format(trainable_count))
    print('\n Non-trainable params: {:,}'.format(non_trainable_count))
    print('\n\n---</summary>---')
    return emonet


def Emonet_extend(num_classes):
    """
    This model is optional and bigger than the previous one. Practically, this model is less useful than the first one
    therefore is not called by the application. Use only for experimental purposes
    """
    emonet = Sequential()

    """
    Convolution and Maxpool layers: Block 1
    """
    # Conv Layer 1:48x48x32
    emonet.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear', input_shape=(48, 48, 1)))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 2:48x48x32
    emonet.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 1
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.2))

    """
    Convolution and Maxpool layers: Block 2
    """
    # Conv Layer 3:24x24x64
    emonet.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 4:24x24x64
    emonet.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 5:24x24x64
    emonet.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 2
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.2))

    """
    Convolution and Maxpool layers: Block 3
    """
    # Conv Layer 6:12x12x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 7:12x12x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 8:12x12x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 9:12x12x256
    emonet.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 3
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.3))

    """
    Convolution and Maxpool layers: Block 4
    """
    # Conv Layer 9:6x6x256
    emonet.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 10:6x6x256
    emonet.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 11:6x6x256
    emonet.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # Conv Layer 12:6x6x256
    emonet.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))

    # MaxPool layer: 4
    emonet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    emonet.add(Dropout(0.2))

    # Flatten
    emonet.add(Flatten())

    # Dense layer 1:
    emonet.add(Dense(2048, activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))
    emonet.add(Dropout(0.5))

    # Dense layer 2:
    emonet.add(Dense(2048, activation='linear'))
    emonet.add(LeakyReLU(alpha=0.3))
    emonet.add(BatchNormalization(axis=-1))
    emonet.add(Dropout(0.5))

    # Output layer
    emonet.add(Dense(num_classes, activation='softmax'))

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(emonet.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(emonet.non_trainable_weights)]))

    # network summary
    print('\n\n---<summary>---')
    print('\n Layers: \n\tConvolution2D: {0}\n\tMaxPooling2D: {1}\n\tFully Connected Layers: {2}'.format(8, 4, 2))
    print('\n Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('\n Trainable params: {:,}'.format(trainable_count))
    print('\n Non-trainable params: {:,}'.format(non_trainable_count))
    print('\n\n---</summary>---')
    return emonet


"""
Using a main function for testing individual modules
Uncomment for testing purposes
Comment when testing is successful 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_emotions", help="Number of emotions in the output layer of the VGG like NN",
                        type=int, default=7, required=True)
    parser.add_argument("-o", "--out_img", type=str, help="Output path to dump the Network Architecture")
    args = parser.parse_args()

    num_emotions = args.num_emotions
    out_img_path = args.out_img

    if num_emotions is not 6 and num_emotions is not 7:
        print("\n Number of emotions options are: \n 6 (for merging anger and disgust) "
              "OR \n 7 (for all the emotions in the dataset")
        exit(0)
    else:
        emonet = Emonet(num_classes=num_emotions)
        emonet.summary()

    if out_img_path is not None:
        plot_model(model=emonet, to_file=out_img_path, show_shapes=True, show_layer_names=True)
