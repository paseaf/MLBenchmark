import gdal
import os, glob
import numpy as np
import sklearn

# open dataset
path = "./image_folder/" # path of the images folder
all_files = glob.glob(os.path.join(path, "*.jpg")) # store path of all .jpg files in the path/ into a list

for file in all_files: # iterate and open each file on the path list
    dataset = gdal.Open(file)  # open the jpg file into gdal dataset
    # read all 3 bands into a numpy array
    ds_arr = np.array(dataset.ReadAsArray()).flatten() # read the dgal dataset into flatted  array
    print(ds_arr)  # print the array
    
print()
