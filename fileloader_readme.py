from fileloader import FileLoader


################################################################################
# This file shows how to prepare the control set and 30 test sets from RGB data
################################################################################

"""I. Prepare control dataset and 30 subsets:"""
# Step 1: create an object of FileLoader.
jpeg_loader = FileLoader(root_path="./Images_RGB/", files_per_class=300)
# print(jpeg_loader.classlist)  # print the list of classes in the folder

# Step 2(optional): set classes to load, otherwise will load all classes in the folder
# jpeg_loader.set_class_list(['AnnualCrop', 'Forest', 'Industrial'])

# Step 3: set control dataset. (self.control_set)
jpeg_loader.set_control_set(num_of_bands=3, is_random=False)    # set up control dataset

# Step 4: set 30 training datasets: from 1% to 50% of whole control dataset
# Subsets are stored in jpeg_loader.training_subsets
jpeg_loader.set_training_subsets(num_of_subsets=30, max_percent=0.5)


"""II. Get data set from a FileLoader instance/object"""
# Get one training set from the FileLoader object
train_set_x, train_set_y = jpeg_loader.training_subsets[0]  # index specifies a subset
print(train_set_x.shape, train_set_y.shape)


# Get the control set from a FileLoader object
control_set_x, control_set_y = jpeg_loader.control_set
print(control_set_x.shape, control_set_y.shape)

"""Others: Utils of FileLoader object"""
# .get_class_name(class_id): convert class id to class name
class_id = train_set_y[0, 3]
print(f'The class name of y= {class_id} is {jpeg_loader.get_class_name(class_id)}')

# .get_class_id(class_name): convert class name to class id
class_name = jpeg_loader.classlist[0]
print(f'The class id of the class \' {class_name}\' is {jpeg_loader.get_class_id(class_name)}')

