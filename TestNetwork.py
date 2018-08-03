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
 " Description: The main network testing class. Tests saved models against the test set to produce top 1% accuracy
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: July 2018
"""
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from BuildDataSet import BuildDataSet
from hdf5datasetgenerator import HDF5DatasetGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import argparse
import os


class EmotionNetworkTest:
    def __init__(self, base_path, num_classes):
        print("\n Testing trained model for emotion recognition of FER 2013 database (TEST SPLIT)")
        self.base_path = base_path
        self.num_classes = num_classes

    def testNetwork(self, trained_model_path, test_dataset_path):
        test_augmentation = ImageDataGenerator(rescale=1/255)
        iap = ImageToArrayPreprocessor()

        # get file details
        config = BuildDataSet(base_path=self.base_path, num_classes=self.num_classes)

        test_generation = HDF5DatasetGenerator(test_dataset_path, 64,
                                               aug=test_augmentation, preprocessors=[iap], classes=config.num_classes)

        # load pre-trained model to test accuracy
        print("\n Loading model: {0}".format(trained_model_path))

        trained_model = load_model(trained_model_path)

        # evaluate model against test set
        print("Evaluate model against test set")
        (test_loss, test_acc) = trained_model.evaluate_generator(test_generation.generator(),
                                                       steps=test_generation.numImages // config.batch_size,
                                                       max_queue_size=config.batch_size * 2)

        print("\n \n FINAL MODEL ACCURACY: {:.2f} %".format(test_acc*100))

        print("\n \n *********************Testing Complete*********************\n")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path",
                        help="The base directory path of the dataset. \n Please ensure that you are"
                             " following the directory structure outlined in the documentation.\n "
                             "Also ensure that the fer2013.csv file is place in the fer2013 folder",
                        type=str, required=True)

    parser.add_argument("-n", "--num_emotions", help="Number of emotions same as dataset build file. \n"
                                                     "If num_emotions = 6 (we merge anger and disgust)\n"
                                                     "if num_emotions = 7 (we use all 7 defined emotions)\n"
                                                     "Default value = 7",
                        type=int, required=True)

    parser.add_argument("-im", "--input_model", help="The name of the pretrained input model to be loaded for test.",
                        type=str, required=True)

    args = parser.parse_args()

    emo_test = EmotionNetworkTest(base_path=args.base_path, num_classes=args.num_emotions)
    emo_test.testNetwork(trained_model_path=os.path.join(args.base_path, 'output', args.input_model),
                         test_dataset_path=os.path.join(args.base_path, 'hdf5', "test.hdf5"))

