# Data structures to store results
class Result:
    """Data structure to store 5 acc_idx_values and test time"""
    def __init__(self):
        self.test_time = None
        self.acc_dict = {'ACC': -1, 'BER': -1, 'CE': -1, 'CRAMERV': -1, 'KAPPA': -1}


class ResultRecorder:
    """Data structure to stores result of 1 train_method on 1 train_set for 3 validation methos"""
    def __init__(self, train_set_size, train_method_name):
        self.train_set_size = train_set_size
        self.train_method_name = train_method_name
        self.train_time = None
        self.results_dict = {'train': None,   # each validation method has a result Object
                             'kfold': None,
                             'all': None}
