# Data structures for plotting


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
    def __init__(self, train_method_name, num_of_files, train_time, test_time):
        self.train_method_name = train_method_name
        self.num_of_files = num_of_files
        self.train_time = train_time
        self.test_time = test_time
