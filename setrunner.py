import utils
from result import Result, ResultRecorder
import time
from plotutils import AccPoint, TimePoint


class SetRunner:
    """Do train and validation for one training subset"""

    def __init__(self, train_set, control_set):
        self.train_set = train_set
        self.control_set = control_set
        self.model_dict = {}
        self.train_size = train_set[1].size
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
            t0 = time.time()
            model = utils.call_train_method(train_method_name, self.train_set)     # train
            t1 = time.time()
            result_recorder.train_time = t1 - t0     # save training time to result
            self.result_recorder_list.append(result_recorder)
            self.model_dict[train_method_name] = model  # save model to model_dict

    def validate_on(self, validation_method, acc_idx_name_list):
        """Validate with ONLY ONE given validation methods"""
        # TODO: implement kfold validation
        if validation_method == 'train' or 'kfold':
            test_set_x, test_set_y = self.train_set
        elif validation_method == 'all':
            test_set_x, test_set_y = self.control_set
        else:
            raise Exception(f'Method "{validation_method}" not supported!')

        for result_recorder in self.result_recorder_list:
            result = Result()   # create a result object for the valid method
            model = self.model_dict[result_recorder.train_method_name]  # get trained model

            # predict
            t0 = time.time()
            y_predict = model.predict(test_set_x)
            t1 = time.time()
            result.test_time = t1 - t0    # save validation time to result

            # run accuracy tests
            for acc_idx_name in self.acc_idx_name_list:
                # TODO: Finish other accuracy indices in utils.py module
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

