import glob
import os
from PIL import Image
import numpy as np
import random


class FileLoader:
    def __init__(self, root_path, files_per_class):
        """
        Must call:
            1. Contructor
            2. set_control_set: set up control dataset
            3. set_training_subsets: set up training subsets

        :param root_path: Path of the parent folder of all classes. E.g. ./2750
        :param files_per_class: Number of files per class.
        """
        self.root_path = root_path  # path of the parent folder of all classes
        self.files_per_class = files_per_class  # number of pictures to take from each folder
        self.classlist = None  # set later using set_class_list
        self.class_data_map = {}  # initialize class->data map ('AnnualCrop' -> (X, Ystring))
        self.num_of_features = 64*64*3   # TODO: later modify the dimension for tif files.
        self.control_set = ()           # (x, y)
        self.training_subsets = []  # [(x1, y1), (x2, y2), ...]

    def set_class_list(self, classlist=None):  # set class list manually
        """
        Specify which classes should be chosen for the test.
        All sub folders will be included by default.

        :param classlist: A list of strings. For example, ['AnnualCrop', 'Forest', ...]
        """
        self.classlist = os.listdir(self.root_path) if classlist is None else classlist

    @staticmethod
    def load_class(class_path: str, files_per_class: int, num_of_bands=3, is_random=False):
        """
        Read a folder of images and return data_set_x, name of folder.

        :param class_path: The path of a folder of images
        :param files_per_class: Number of files to load from the folder
        :param num_of_bands: Number of bands in the images
        :param is_random: If the files should be chosen randomly
        :return: An nxm numpy array of samples, a string label
        """
        num_of_features = num_of_bands * 64 * 64
        control_set_x = np.zeros((files_per_class, num_of_features))
        control_set_label = os.path.basename(class_path)  # one string
        file_names = glob.glob(os.path.join(class_path, "*.jpg"))

        # shuffle the list of files if necessary
        if is_random:
            random.shuffle(file_names)  # ransom.shuffle works in place!
        # read files to np array
        for i in range(files_per_class):
            with open(file_names[i], 'rb') as file_stream:
                pixels = np.array(Image.open(file_stream)).flatten() / 255.0  # normalization
                mean = pixels.mean()
                pixels -= mean  # data centering
                        # https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
                control_set_x[i, :] = pixels  # write image vector to the i-th row of X
        return control_set_x, control_set_label

    def set_control_set(self, num_of_bands=3, is_random=False):
        """Set up control dataset self.control_set"""
        control_set_x = np.empty((0, self.num_of_features))
        control_set_y = np.empty(0, dtype=np.int8)
        # open folder of each class and read X and Y
        for class_name in self.classlist:
            # set up class_data_map
            class_path = os.path.join(self.root_path, class_name)
            class_set_x, class_label = self.load_class(class_path, self.files_per_class, num_of_bands, is_random)
            self.class_data_map[class_label] = class_set_x
            # set up control_set
            control_set_x = np.concatenate((control_set_x, class_set_x), axis=0)
            control_set_y = np.concatenate((control_set_y, np.full(self.files_per_class, self.get_class_id(class_name))))
        self.control_set = (control_set_x, control_set_y)

    def set_training_subsets(self, num_of_subsets=30, max_percent=0.5):
        """set up self.training_subsets: a list of (train_data_x, train_data_y)"""
        if self.class_data_map == {}:
            self.set_control_set()   # set controlset map with default settings
        # get a list of number of samples in each train set
        tmp = np.arange(1, num_of_subsets + 1)  # generata a list of [1, ..., num_of_subsets]
        list_of_percents = tmp / num_of_subsets * max_percent  # get the list of percentages of samples in each subsets
        list_of_sizes = np.around(list_of_percents * self.files_per_class, decimals=0).astype(int)  # get a list of number of files in each subsets
        # create train sets
        training_subsets = []
        for samples_per_class in list_of_sizes:
            train_set_x, train_set_y = self.__get_train_set(samples_per_class)
            training_subsets.append((train_set_x, train_set_y))
        self.training_subsets = training_subsets

    def __get_train_set(self, samples_per_class):
        """Return (x, y) for given number of samples per class"""
        rows_idx = np.random.randint(0, self.files_per_class, size=samples_per_class)  # list of indices to select from each class
        # create matrix for train set
        (m, n) = (len(self.classlist) * samples_per_class, self.num_of_features)
        train_set_x = np.empty((0, n))  # TODO: specify dtype for x.
        train_set_y = np.empty(0, dtype=np.int8)
        # col_count = 0
        y_count = 0
        # 1. iterate through X of each class
        # 2. select columns with idx
        # 3. write the selected data to train_set_x
        listy = list(self.class_data_map.values())
        for x_of_one_class in listy:  # iterate through X of each class
            selected_x = x_of_one_class[rows_idx, :]
            train_set_x = np.concatenate((train_set_x, selected_x), axis=0)
            train_set_y = np.concatenate((train_set_y, np.full(samples_per_class, y_count)))
            y_count += 1
        return train_set_x, train_set_y

    def get_class_name(self, class_int: int):
        """convert class id to class name"""
        return self.classlist[class_int]

    def get_class_id(self, class_name: str):
        """convert class name to class id"""
        return self.classlist.index(class_name)
