from testrunner import TestRunner
from fileloader import FileLoader

"""Load data"""
# file_path = '/home/ole/Documents/Informatik/SS19/DBPRO/EuroSAT/2750'  # Ole
file_path = './Images_RGB/'
jpeg_loader = FileLoader(root_path=file_path, files_per_class=300)
jpeg_loader.set_class_list(['AnnualCrop', 'Forest', 'Industrial'])
jpeg_loader.set_control_set(num_of_bands=3, is_random=False)    # set up control dataset
jpeg_loader.set_training_subsets(num_of_subsets=30, max_percent=0.5)

"""Run test and validation"""
test_runner = TestRunner(jpeg_loader)   # set test_runner for running test
test_runner.init_all()  # initialize set_runner for each training subset

# Check utils.py for Names of train methods, validation methods and accuracy indices
test_runner.train_all(train_method_list=['lda'])    # train all subset with given mlm in train_method_list
test_runner.validate_all(validation_method_list=['train', 'all'], acc_idx_name_list=['ACC', 'BER'])  # do validation

test_runner.set_all_points()    # set acc_points and time points for plotting

"""Get list of points for plotting"""
acc_point_list = test_runner.get_acc_points()  # a list of AccPoint objects, each represents one point
# Number of acc points = #train_subset * #train_method * #acc_idx * #validation_method = 30 * 1 * 2 * 2 = 120

time_point_list = test_runner.get_time_points()    # a list of TimePoints, for the other plotting
# Number of time points = #train_subset * #train_method * #validation_method = 30 * 1 * 2 = 60

# TODO: Convert AccPoints objects to tuple when necessary

"""Plot"""
# TODO: plot







