# This script creates a raster file with the topological complexity of the areas of interest

import numpy as np
import gdal
import ogr
import rasterio

# Defining relevant functions
def rasterize(data, vectorSrc, field, outFile, value_nodata=-9999):
    """Adapted from: https://github.com/nkarasiak/dzetsaka
    This function rasterize a vector using a raster as a model.

    Args:
        data (str): Location of the raster used as model. 
        vectorSrc (str): Location of the vector that will be rasterised.
        field (str): Name of the field in the vector file that will be burned as values in the raster file.
        outFile (str): Name of the output file (include .tif in the name).
        value_no_data (int): Value that should be used in the raster in areas not covered by vector data. 

    Returns:
        str: Location of the output file (raster file).
    """
    dataSrc = gdal.Open(data)
    shp = ogr.Open(vectorSrc)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        outFile,
        dataSrc.RasterXSize,
        dataSrc.RasterYSize,
        1,
        gdal.GDT_UInt16)
    dst_ds.SetGeoTransform(dataSrc.GetGeoTransform())
    dst_ds.SetProjection(dataSrc.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(value_nodata)
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], lyr, None)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, options=OPTIONS)

    data, dst_ds, shp, lyr = None, None, None, None
    return outFile

# Defining relevant variables
PATH_RASTER = 'C:/Users/agati/OneDrive - University College Dublin/Datasets/20200714 FrontierDevelopmentLab/LowResolution/'
raster_capetown = 'Capetown.tif'
raster_mumbai = 'Mumbai.tif'

PATH_COMPLEXITY = 'C:/Users/agati/Documents/DatasetsLocal/TopologicalAnalysis/'
file_south_africa = 'ZAF.geojson'
file_india = 'IND.geojson'
outFile_capetown = PATH_COMPLEXITY + 'capetown_complexity.tif'
outFile_mumbai = PATH_COMPLEXITY + 'mumbai_complexity.tif'

# Generating raster files
output_capetown = rasterize(data=PATH_RASTER+raster_capetown, vectorSrc=PATH_COMPLEXITY+file_south_africa, field='complexity', outFile=outFile_capetown, value_nodata=99)
complexity_capetown = rasterio.open(outFile_capetown)
print(np.unique(complexity_capetown.read(), return_counts=True))

output_mumbai = rasterize(data=PATH_RASTER+raster_mumbai, vectorSrc=PATH_COMPLEXITY+file_india, field='complexity', outFile=outFile_mumbai, value_nodata=99)
complexity_mumbai = rasterio.open(outFile_mumbai)
print(np.unique(complexity_mumbai.read(), return_counts=True))