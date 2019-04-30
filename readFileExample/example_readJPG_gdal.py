import gdal
import numpy as np
# open dataset
src_ds = gdal.Open('AnnualCrop_1.jpg') # change AnnualCrop_1.jpg to the location of your directory/file

# read band 1 into a numpy array
myarray = np.array(src_ds.GetRasterBand(1).ReadAsArray())
print(myarray) # print the array
print(myarray.shape) # print the shape of the array

# read three bands iterately
print("\n[ RASTER BAND COUNT ]: %d " % src_ds.RasterCount)
for band in range(src_ds.RasterCount):
    band += 1
    print("[ GETTING BAND ]: %d" % band)
    srcband = src_ds.GetRasterBand(band)
    if srcband is None:
        continue

    stats = srcband.GetStatistics(True, True)
    if stats is None:
        continue

    print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % (
                stats[0], stats[1], stats[2], stats[3]))