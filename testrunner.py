from setrunner import SetRunner


class TestRunner:
    """Do train and validation for a FileLoader object / on all training subsets"""
    def __init__(self, fileloader):
        self.fileloader = fileloader
        self.set_runner_list = []

    def init_all(self):

        for train_set in self.fileloader.training_subsets:
            set_runner = SetRunner(train_set, self.fileloader.control_set)
            self.set_runner_list.append(set_runner)

    def train_all(self, train_method_list=None):
        """Train on every training subset"""
        train_method_list = ['lda'] if train_method_list is None else train_method_list
        for set_runner in self.set_runner_list:
            set_runner.train_all_models(train_method_list)

    def validate_all(self, validation_method_list=None, acc_idx_name_list=None):  # TODO: add 'kfold' after implementation
        """Validate on every training subset"""
        validation_method_list = ['train', 'all'] if validation_method_list is None else validation_method_list
        acc_idx_name_list = ['ACC', 'BER'] if acc_idx_name_list is None else acc_idx_name_list
        for set_runner in self.set_runner_list:
            set_runner.validate(validation_method_list, acc_idx_name_list)

    def set_all_points(self):
        for set_runner in self.set_runner_list:
            set_runner.set_all_points()

    def get_acc_points(self):
        # Return a list of AccPoint objects, each represents one point
        result = []
        for set_runner in self.set_runner_list:
            result += set_runner.acc_point_list
        return result

    def get_time_points(self):
        # Return a list of TimePoints objects, each represents one point
        result = []
        for set_runner in self.set_runner_list:
            result += set_runner.time_point_list
        return result
