# Data structures for plotting
from collections import namedtuple
from readFileExample.plotting import plot

class AccPoint:
    """Each object represents a point in accuracy metrics diagram"""
    def __init__(self, valid_method_name, acc_idx_name, train_method_name, num_of_files, acc_idx_value):
        self.valid_method_name = valid_method_name
        self.acc_idx_name = acc_idx_name
        self.train_method_name = train_method_name
        self.num_of_files = num_of_files
        self.acc_idx_value = acc_idx_value


class TimePoint:
    """Each object represents a point in time metrics diagram"""
    def __init__(self, train_method_name, validation_method, num_of_files, train_time, test_time):
        self.train_method_name = train_method_name
        self.validation_method = validation_method
        self.num_of_files = num_of_files
        self.train_time = train_time
        self.test_time = test_time


PlotData = namedtuple('PlotData', ['X', 'Y'])


class PlotHelper:
    def __init__(self, acc_points, time_points):
        self.acc_points = acc_points
        self.time_points = time_points
        self.acc_results = {}
        self.time_results = {}
        self.acc_idx_list = []
        self.train_method_list =[]

    def set_results(self):
        self.set_acc_results()
        self.set_time_results()

    def set_acc_results(self):
        """
        {'valid_method_name':{
                (train_method, acc_idx): (X, Y) # X: list of #files
                                                # Y: list of accuracy values
            }
        }
        """
        for acc_point in self.acc_points:
            valid_method = acc_point.valid_method_name
            if valid_method not in self.acc_results:
                self.acc_results[valid_method] = {}  # add entry to the dict

            train_method, acc_idx = acc_point.train_method_name, acc_point.acc_idx_name
            if train_method not in self.train_method_list:
                self.train_method_list.append(train_method)
            if acc_idx not in self.acc_idx_list:
                self.acc_idx_list.append(acc_idx)

            if (train_method, acc_idx) not in self.acc_results[valid_method]:  # add entry
                self.acc_results[valid_method][(train_method, acc_idx)] = PlotData([], [])

            self.acc_results[valid_method][(train_method, acc_idx)].X.append(acc_point.num_of_files)
            self.acc_results[valid_method][(train_method, acc_idx)].Y.append(acc_point.acc_idx_value)

    def set_time_results(self):
        """
        {'valid_method_name/train':{
                'train_method': (X, Y)  # X: list of #files
                                        # Y: list of time
            }
        }
        """
        tmp_valid_name = ''  # auxiliary variable to record training time
        self.time_results['training'] = {}
        for time_point in self.time_points:
            valid_method, train_method = time_point.validation_method, time_point.train_method_name
            if valid_method not in self.time_results:
                self.time_results[valid_method] = {}  # add entry to the dict
                tmp_valid_name = tmp_valid_name or valid_method  # set to current valid method name if not set yet

            if valid_method == tmp_valid_name:  # store data to time_results['train']
                if train_method not in self.time_results['training']:
                    self.time_results['training'][train_method] = PlotData([], [])
                self.time_results['training'][train_method].X.append(time_point.num_of_files)
                self.time_results['training'][train_method].Y.append(time_point.train_method_name)

            if train_method not in self.time_results[valid_method]: # add entry for one mlm
                self.time_results[valid_method][train_method] = PlotData([], [])

            self.time_results[valid_method][train_method].X.append(time_point.num_of_files)
            self.time_results[valid_method][train_method].Y.append(time_point.test_time)

    def get_valid_methods(self):
        return list(self.acc_results.keys())

    def get_train_acc_pairs(self):
        valid_name = list(self.acc_results.keys())[0]
        return list(self.acc_results[valid_name].keys())

    def get_ml_methods(self):
        valid_name = list(self.acc_results.keys())[0]
        return list(self.time_results[valid_name].keys())

    def get_time_plot_names(self):
        return list(self.time_results.keys())

    def plot_acc(self):
        for valid_method in list(self.acc_results.keys()):
            plot(self.acc_results[valid_method], self.acc_idx_list, self.train_method_list, valid_method)
