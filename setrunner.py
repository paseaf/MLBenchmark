import utils
from result import Result, ResultRecorder
import time
from plotutils import AccPoint, TimePoint
from sklearn.model_selection import KFold


class SetRunner:
    """Do train and validation for one training subset"""

    def __init__(self, train_set, control_set):
        self.train_set = train_set
        self.control_set = control_set
        self.model_dict = {}
        self.train_size = train_set[1].size  # number of samples in this training set
        self.result_recorder_list = []     # list of ResultRecorder Objects
        self.acc_point_list = []    #
        self.time_point_list = []
        self.validation_method_list = None  # set on calling  validate()
        self.acc_idx_name_list = None   # set on calling validate()

    def train_all_models(self, train_method_list):
        """Train all models on the data set"""
        for train_method_name in train_method_list:
            result_recorder = ResultRecorder(self.train_size, train_method_name)
            # train model
            print(f'Training with {train_method_name} on {self.train_size} files...')
            t0 = time.time()
            model = utils.call_train_method(train_method_name, self.train_set)     # train
            t1 = time.time()
            result_recorder.train_time = t1 - t0     # save training time (seconds) to result
            print(f'Training finished in {result_recorder.train_time} seconds.')
            self.result_recorder_list.append(result_recorder)
            self.model_dict[train_method_name] = model  # save model to model_dict

    def validate_on(self, validation_method, acc_idx_name_list):
        """Validate with ONLY ONE given validation methods"""
        # TODO: implement kfold validation
        if validation_method in ['train', 'kfold']:
            test_set_x, test_set_y = self.train_set
        elif validation_method == 'all':
            test_set_x, test_set_y = self.control_set
        else:
            raise Exception(f'Method "{validation_method}" not supported!')

        for result_recorder in self.result_recorder_list:
            result = Result()   # create a result object for the valid method
            model = self.model_dict[result_recorder.train_method_name]  # get trained model

            # predict
            print(f'Testing model {result_recorder.train_method_name} with {validation_method} '
                  f'validation method on {test_set_y.size} files...')
            if validation_method == 'kfold':
                k = min(test_set_y.size, 10)  # k=10 if #test_examples > 10
                kf = KFold(n_splits=k)
                # kf.get_n_splits(test_set_x)
                temp_acc_dict = {}  # temp acc dict to store SUM of acc values for each acc idx
                t0 = time.time()
                for train_index, test_index in kf.split(test_set_x):  # iterate through k folds
                    X_test_fold = test_set_x[test_index]  # get the k-th fold from the test dataset
                    y_test_fold = test_set_y[test_index]
                    y_predict_fold = model.predict(X_test_fold)

                    # run accuracy tests for one fold
                    for acc_idx_name in acc_idx_name_list:
                        if acc_idx_name not in temp_acc_dict:
                            temp_acc_dict[acc_idx_name] = 0
                        temp_acc_dict[acc_idx_name] += utils.call_acc_idx_method(acc_idx_name, y_test_fold, y_predict_fold)

                t1 = time.time()
                result.test_time = t1 - t0  # save validation time to result
                print(f'Testing finished in {result.test_time} seconds.')

                # get avg acc value
                for acc_idx_name in acc_idx_name_list:
                    result.acc_dict[acc_idx_name] = temp_acc_dict[acc_idx_name] / k  # get average
            else:
                t0 = time.time()
                y_predict = model.predict(test_set_x)
                t1 = time.time()
                result.test_time = t1 - t0    # save validation time to result
                print(f'Testing finished in {result.test_time} seconds.')

                # run accuracy tests
                for acc_idx_name in self.acc_idx_name_list:
                    result.acc_dict[acc_idx_name] = utils.call_acc_idx_method(acc_idx_name, test_set_y, y_predict)

            # save result of current validation method to dict
            result_recorder.results_dict[validation_method] = result

    def validate(self, validation_method_list, acc_idx_name_list):
        """Validate with all given validation methods"""
        self.validation_method_list = validation_method_list
        self.acc_idx_name_list = acc_idx_name_list
        for validation_method in validation_method_list:
            self.validate_on(validation_method, acc_idx_name_list)

    def set_all_points(self):
        """Set result points for plotting"""
        for result_recorder in self.result_recorder_list:
            num_of_files = result_recorder.train_set_size
            train_method_name = result_recorder.train_method_name
            train_time = result_recorder.train_time

            # TODO: implement kfold validation
            for validation_method in self.validation_method_list:
                result = result_recorder.results_dict[validation_method]
                test_time = result.test_time
                time_point = TimePoint(train_method_name, validation_method, num_of_files, train_time, test_time)
                self.time_point_list.append(time_point)

                for acc_idx_name in self.acc_idx_name_list:
                    acc_idx_value = result.acc_dict[acc_idx_name]
                    acc_point = AccPoint(validation_method, acc_idx_name, train_method_name, num_of_files,
                                         acc_idx_value)
                    self.acc_point_list.append(acc_point)

