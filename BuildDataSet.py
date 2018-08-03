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
 " Description: Core Module: Process
                ====================================================================================
                Module Description:
                1) Load Kaggle dataset in the form of a CSV file
                2) Parse Kaggle CSV file
                3) Save dataset into HDF5 files for training, validation and test sets
                ====================================================================================
 " Author: Ratnajit Mukherjee, ratnajitmukherjee@gmail.com
 " Date: July 2018
"""

# various imports
import os
import numpy as np
import argparse
from hdf5datasetwriter import HDF5DatasetWriter


class BuildDataSet:
    def __init__(self, base_path, num_classes):
        print("\n Base Path: {0}".format(base_path))

        self.input_path = os.path.join(base_path, 'fer2013')

        # directory structure check
        if not os.path.exists(self.input_path):
            print("\n Input directory structure does not exist. Manually create the directory structure "
                  "following documentation")
            exit(-1)

        # directory structure check
        self.hdf5_path = os.path.join(base_path, 'hdf5')
        if not os.path.exists(self.hdf5_path):
            print("\n Uncompressed data directory structure does not exist. Manually create the HDF5 directory" 
                  "following documentation")
            exit(-1)

        # directory structure check
        self.output_path = os.path.join(base_path, 'output')
        if not os.path.exists(self.output_path):
            print("\n Output directory structure does not exist. Manually create the output directory" 
                  "following documentation")
            exit(-1)

        # define number of classes
        self.num_classes = num_classes  # set to 6 if you are ignoring the 'disgust' class

        # define the batch size
        self.batch_size = 64

        print("\n Input path: {0}\n Intermediate HDF5 path: {1}\n Output HDF5 path: {2}\n # of Emotions: {3}"
              .format(self.input_path, self.hdf5_path, self.output_path, self.num_classes))

    def config_dataset(self):
        input_csv_file = os.path.join(self.input_path, 'fer2013.csv')
        train_HDF5 = os.path.join(self.hdf5_path, 'train.hdf5')
        val_HDF5 = os.path.join(self.hdf5_path, 'val.hdf5')
        test_HDF5 = os.path.join(self.hdf5_path, 'test.hdf5')

        # check if the csv file is properly placed or not
        if not os.path.isfile(input_csv_file):
            print("\nThe FER2013 dataset in .csv format was not found. Please manually place that file in the directory")
            exit(-1)

        print('\n Input dataset file: {0}\n Train dataset: {1}\n Validate dataset: {2}\n Test dataset: {3}'
              .format(input_csv_file, train_HDF5, val_HDF5, test_HDF5))

        return input_csv_file, train_HDF5, val_HDF5, test_HDF5

    def build_dataset(self, input_csv_file, train_HDF5, val_HDF5, test_HDF5):
        print("\n [STATUS: ] Loading data.... Please wait")

        # Open the Kaggle dataset input CSV file
        input_file = open(input_csv_file)
        input_file.__next__()

        # initiate the training, validation and test data sets (empty)
        (trainImages, trainLabels) = ([], [])
        (valImages, valLabels) = ([], [])
        (testImages, testLabels) = ([], [])

        # loop over each of the input file
        for row in input_file:
            # extract the label, image, and usage from the row
            (label, image, usage) = row.strip().split(",")
            label = int(label)

            # We are going to ignore the disgust label and merge them with angry (refer Memong paper)
            if self.num_classes == 6:
                # merge together the "anger" and "disgust classes
                if label == 1:
                    label = 0

                # if label has a value greater than zero, subtract one from
                # it to make all labels sequential (not required, but helps
                # when interpreting results)
                if label > 0:
                    label -= 1

            # reshape the flattened pixel list into a 48x48 (grayscale) image
            image = np.array(image.split(" "), dtype=np.uint8)
            image = image.reshape((48, 48))

            """
            ===============================================================================================
            Splitting the data into train, validation and test set based on the usage given in the CSV file
            NOTE: Validation is noted as PrivateTest in the CSV file 
            ===============================================================================================
            """

            # check if usage = Training
            if usage == "Training":
                trainImages.append(image)
                trainLabels.append(label)

            # check if usage = Validation
            elif usage == "PrivateTest":
                valImages.append(image)
                valLabels.append(label)

            # check if usage = "Test"
            elif usage == "PublicTest":
                testImages.append(image)
                testLabels.append(label)

        # list pair for training, validation and test sets along with their corresponding files
        datasets = [(trainImages, trainLabels, train_HDF5), (valImages, valLabels, val_HDF5),
                    (testImages, testLabels, test_HDF5)]

        for (images, labels, dataset_path) in datasets:
            # check if file exists
            if os.path.isfile(dataset_path):
                print('File {0} already exists. Skipping...'.format(dataset_path))
                continue
            else:
                # create HDF5 writer
                print("\n [STATUS: ] Building and Writing {0}...".format(dataset_path))
                writer = HDF5DatasetWriter((len(images), 48, 48), dataset_path)

                # loop over each image and add them to each of the dataset files
                for (image, label) in zip(images, labels):
                    writer.add([image], [label])

                writer.close()

        input_file.close()
        return


"""
Using a main function for testing individual modules
Uncomment for testing purposes
Comment when testing is successful 
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--base_path", help="[Required] The base directory path of the dataset. \n Please ensure that"
                        " you are following the directory structure outlined in the documentation. \n Also ensure that "
                        "the fer2013.csv file is place in the fer2013 folder",
                        type=str, required=True)
    parser.add_argument("-n", "--num_emotions", help="[Required] Set number of emotions used to build the datasets. \n"
                                                     "If num_emotions = 6 (we merge anger and disgust)\n"
                                                     "if num_emotions = 7 (we use all 7 defined emotions)\n"
                                                     "Default value = 7",
                        type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    if not os.path.exists(base_path):
        print("\n Base path does not exist. Kindly follow the documentation and manually create it.")
        exit(0)

    num_classes = args.num_emotions
    bds = BuildDataSet(base_path, num_classes)
    (input_csv_file, train_HDF5, val_HDF5, test_HDF5) = bds.config_dataset()
    bds.build_dataset(input_csv_file, train_HDF5, val_HDF5, test_HDF5)
