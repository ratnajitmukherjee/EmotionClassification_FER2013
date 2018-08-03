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
 " Description: The main network training class. Contains the following submodules
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: July 2018
"""

# various imports required to precess, data_augment and train the Network
from keras.callbacks import EarlyStopping, ModelCheckpoint
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K
from hdf5datasetgenerator import HDF5DatasetGenerator
from BuildDataSet import BuildDataSet
from VGGNet import Emonet
import matplotlib.pyplot as plt
import argparse
import os


class NetworkTrainingMain:
    def __init__(self, base_path, num_classes):
        self.base_path = base_path
        self.num_classes = num_classes

    def model_plot_history(self, emotion_train):
        plt.plot(emotion_train.history['acc'], 'r+',  linestyle='-', label='Training accuracy')
        plt.plot(emotion_train.history['loss'], 'b+',  linestyle='-.', label='Training loss')

        plt.plot(emotion_train.history['val_acc'], 'rx', linestyle='-', label='Validation accuracy')
        plt.plot(emotion_train.history['val_loss'], 'bx', linestyle='-.', label='Validation loss')
        plt.minorticks_on()
        plt.ylabel("Model Training History")
        plt.xlabel("Epochs")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()
        return

    def train_dataset(self, num_classes, pretrained_model_name, new_model_name, new_learning_rate, num_epochs):
        # calling other supporting classes to get the training
        config = BuildDataSet(self.base_path, num_classes)   # getting the constructor variables
        (input_csv_file, train_HDF5, val_HDF5, test_HDF5) = config.config_dataset()     # getting the returned file

        # construction template for training and validation data augmentation using keras functions

        training_data_augmentation = ImageDataGenerator(rotation_range=25, zoom_range=0.5, horizontal_flip=True,
                                                        rescale=(1/255))
        validation_data_augmentation = ImageDataGenerator(rescale=(1/255))

        # Initialize image to array preprocessor class used by Adrian's HDF5 data generator
        iap = ImageToArrayPreprocessor()

        # Now using Adrian's function for data generation
        training_generator = HDF5DatasetGenerator(train_HDF5, config.batch_size, aug=training_data_augmentation,
                                                  preprocessors=[iap], classes=config.num_classes)

        validation_generator = HDF5DatasetGenerator(val_HDF5, config.batch_size, aug=validation_data_augmentation,
                                                    preprocessors=[iap], classes=config.num_classes)

        if pretrained_model_name is None:
            # Compile model and start training from EPOCH 1

            # set Adam Optimizer to default rate
            opt = Adam(lr=1e-3)
            emo_model = Emonet(config.num_classes)
            # emo_model = Emonet_extend(config.num_classes)
            emo_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        else:
            emo_model = load_model(pretrained_model_name)
            if new_learning_rate is None:
                old_learning_rate = K.get_value(emo_model.optimizer.lr)
                new_learning_rate = old_learning_rate / 10
                K.set_value(emo_model.optimizer.lr, new_learning_rate)
            else:
                old_learning_rate = K.get_value(emo_model.optimizer.lr)
                K.set_value(emo_model.optimizer.lr, new_learning_rate)

            print("\n Changing learning rate from {0} to {1}".format(old_learning_rate, new_learning_rate))

        # list of keras callbacks
        checkpoint_filepath = os.path.join(config.output_path, "emotion_weights-{epoch:02d}.hdf5")
        emotion_callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                           ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, period=5)]

        # check number of epochs
        if num_epochs is None:
            num_epochs = 50

        print('\n\n*************TRAINING START*******************\n')

        emotion_train = emo_model.fit_generator(training_generator.generator(),
                                                steps_per_epoch=training_generator.numImages/config.batch_size,
                                                validation_data=validation_generator.generator(),
                                                validation_steps=validation_generator.numImages/config.batch_size,
                                                epochs=num_epochs, max_queue_size=config.batch_size*2,
                                                callbacks=emotion_callbacks)

        # close the training and validation generators
        training_generator.close()
        validation_generator.close()

        emo_model.save(filepath=os.path.join(config.output_path, new_model_name))

        print('\n\n*************TRAINING COMPLETE**********************\n')

        self.model_plot_history(emotion_train)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", help="The base directory path of the dataset. \n Please ensure that you are"
                                                  " following the directory structure outlined in the documentation.\n "
                                                  "Also ensure that the fer2013.csv file is place in the fer2013 folder",
                        type=str, required=True)

    parser.add_argument("-n", "--num_emotions", help="Set number of emotions equal to the dataset build file.\n"
                                                     "If num_emotions = 6 (we merge anger and disgust)\n"
                                                     "if num_emotions = 7 (we use all 7 defined emotions)\n"
                                                     "Default value = 7",
                        type=int, required=True)

    parser.add_argument("-im", "--input_model", help="The name of the pretrained input model to be loaded if available."
                                                     "\n If not then it defaults to None..", type=str)
    parser.add_argument("-om", "--output_model", help="The name of final model which will be saved. \n If input is None"
                                                      " then default name is used.", type=str)
    parser.add_argument("-lr", "--learning_rate", help="The learning rate of the training model", default=None,
                        type=float)
    parser.add_argument("-epochs", "--number_of_epochs", help="Enter the number of epochs that the model is to trained"
                                                              "and/or fine-tuned", type=int, required=True)

    args = parser.parse_args()

    # parsing the values passed:
    base_path = args.base_path
    num_classes = args.num_emotions

    if args.input_model is not None:
        input_model_name = os.path.join(base_path, 'output', args.input_model)
    else:
        input_model_name = None

    if args.output_model is not None:
        output_model_name = args.output_model
    else:
        output_model_name = "emotion_classification_final.hdf5"

    if args.learning_rate is not None:
        lr = args.learning_rate
    else:
        print("\n [LEARNING RATE INFO:] \n LR = 0.001 if there is no default model.\n If there is an input model"
              " the program will automatically retrieve the LR and New LR  will be set to Old LR / 10")
        lr = None

    num_epochs = args.number_of_epochs

    train_main = NetworkTrainingMain(base_path, num_classes)
    train_main.train_dataset(num_classes=num_classes, pretrained_model_name=input_model_name,
                             new_model_name=output_model_name,new_learning_rate=lr, num_epochs=num_epochs)
