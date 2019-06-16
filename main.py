from testrunner import TestRunner
from fileloader import FileLoader
from plotutils import PlotHelper


def main(file_path: str, files_per_class: int, mlms: [str] = None, classes: [str] = None, num_of_subsets=10, max_percent=0.5,
         is_random=True, valid_methods: [str] = None):
    """
    Run the test on given files with giving ML algorithms.
    Plot the performance and save the plots in current folder.

    :param file_path: Path of the parent folder of all classes. E.g. ./2750
    :param files_per_class: Number of files per class.
    :param mlms: list of ml algorithms
    :param classes: list of folder names
    :param num_of_subsets: number of files to open per folder
    :param max_percent: max percentage of opened files for training
    :param is_random: select file randomly
    :param valid_methods: list of validation methods
    """
    """Load data"""

    file_large = FileLoader(root_path=file_path, files_per_class=files_per_class)
    classes = ['AnnualCrop', 'Forest', 'SeaLake', 'Pasture', 'HerbaceousVegetation', 'Residential'] if classes is None else classes
    file_large.set_class_list(classes)
    file_large.set_control_set(num_of_bands=3, is_random=is_random)    # set up control dataset
    file_large.set_training_subsets(num_of_subsets=num_of_subsets, max_percent=max_percent)

    """Run test and validation"""
    test_runner = TestRunner(file_large)   # set test_runner for running test
    test_runner.init_all()  # initialize set_runner for each training subset

    # Check utils.py for Names of train methods, validation methods and accuracy indices
    if mlms is None:
        mlms = ['lda', 'knn', 'svm', 'randomForest', 'lr', 'mlp', 'mlpe', 'ct', 'b']
    test_runner.train_all(train_method_list=mlms)    # train all subset with given mlm in train_method_list
    valid_methods = ['train', 'all'] if valid_methods is None else valid_methods
    test_runner.validate_all(validation_method_list=valid_methods)  # do validation with all acc idx

    test_runner.set_all_points()    # set acc_points and time points for plotting

    """Get list of points for plotting"""
    acc_point_list = test_runner.get_acc_points()  # a list of AccPoint objects, each represents one point
    # Number of acc points = #train_subset * #train_method * #acc_idx * #validation_method = 30 * 1 * 2 * 2 = 120

    time_point_list = test_runner.get_time_points()    # a list of TimePoints, for the other plotting
    # Number of time points = #train_subset * #train_method * #validation_method = 30 * 1 * 2 = 60

    """Prepare """
    data = PlotHelper(acc_point_list, time_point_list)
    data.set_results()

    print(f'Validation methods: {data.get_valid_methods()}')
    print(f'(train_method, acc_idx) pairs: {data.get_train_acc_pairs()}')
    print(f'Machine learning methods: {data.get_ml_methods()}')
    print(f'Names of time plots: {data.get_time_plot_names()}')

    print('test')

    """Plot"""

    data.plot_acc()


if __name__ == "__main__":
    # path = '/home/ole/Documents/Informatik/SS19/DBPRO/EuroSAT/2750'  # Ole
    path = '../Images_RGB_full/'
    #path = 'C:\Users\sechs\Downloads\EuroSAT\2750'
    main(file_path=path, files_per_class=100, classes=['AnnualCrop', 'Forest', 'SeaLake', 'Pasture'],
         mlms=['lda', 'knn', 'randomForest', 'lr', 'mlp', 'ct'])
