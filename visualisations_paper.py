# This script produces the graphs shown in the paper

import numpy as np
import pandas as pd
import rasterio
import useful_functions as my_functions

PATH_local = 'C:/Users/agati/OneDrive - University College Dublin/Datasets/20200714 FrontierDevelopmentLab/LowResolution/'
PATH_RESULTS = 'results/'
PATH_OUTPUT = 'results_images/'

list_to_process = ['Mumbai','Capetown']

for location in list_to_process:

    # Setting variables
    tile_height=20
    tile_width=20

    # Reading the ground-truth data
    file_ground_truth = location + '_ground_truth' + '.tif'
    path_with_gt_file = PATH_local+file_ground_truth
    original_raster = rasterio.open(path_with_gt_file)
    original_numpy_array = original_raster.read(1)
    image_reduced = my_functions.reduce_image_for_irregular_tile_cutting(image_array=original_numpy_array,tile_height=tile_height,tile_width=tile_width)
    df, tile_code_array = my_functions.generating_df_and_code_array(image_array=image_reduced,tile_height=tile_height,tile_width=tile_width,threshold=0.5)

    # Reading the unsupervised learning results
    file_results_path = PATH_RESULTS + 'df_features_all_dataset_' + location + '.csv'
    unsupervised_results = pd.read_csv(file_results_path)

    # Merging the ground-truth data and unsupervised learning results
    merged_info = pd.merge(df.loc[:,['tile_ID','code_image_i','code_image_j']],unsupervised_results.loc[:,['tile_ID','cluster_ID']], on='tile_ID', how='left')

    # Populating array with the cluster number in the unsupervised learning results
    cluster_array_reduced = np.zeros((image_reduced.shape[0],image_reduced.shape[1]))
    for i in range(len(merged_info)):
        i_start = merged_info['code_image_i'].iloc[i]*20
        i_end = i_start+20
        j_start = merged_info['code_image_j'].iloc[i]*20
        j_end = j_start+20
        cluster_array_reduced[i_start:i_end,j_start:j_end] = merged_info['cluster_ID'].iloc[i]

    # Double checking assignment of clusters
    df1 = pd.DataFrame(merged_info['cluster_ID'].value_counts(dropna=False))
    df1 = df1.reset_index()
    df1.columns = ['cluster_ID','counts'] 
    values, counts = np.unique(cluster_array_reduced,return_counts=True)
    df2 = pd.DataFrame([values,counts/400], dtype=int)
    df2 = df2.T
    df2.columns = ['cluster_ID','counts'] 
    merged_dfs_check = pd.merge(df1,df2, on='cluster_ID', how='left')
    merged_dfs_check

    # Array with same size as ground-truth
    cluster_array_complete = np.zeros((original_numpy_array.shape[0],original_numpy_array.shape[1]))
    cluster_array_complete[0:cluster_array_reduced.shape[0],0:cluster_array_reduced.shape[1]] = cluster_array_reduced

    # Generating raster with cluster numbers
    raster_cluster = 'raster_clusters_{}_for_paper.tif'.format(location)
    path_new_raster = PATH_OUTPUT + raster_cluster

    with rasterio.open(
    path_new_raster,
    'w',
    driver='GTiff',
    height=cluster_array_complete.shape[0],
    width=cluster_array_complete.shape[1],
    count=1,
    dtype=cluster_array_complete.astype(rasterio.float32).dtype,
    crs=original_raster.crs,
    transform=original_raster.transform
    ) as new_dataset:
        new_dataset.write(cluster_array_complete, 1)

    # Checking the new created file
    # new_raster = rasterio.open(raster_cluster)
    # new_raster_array = new_raster.read(1)
    # np.unique(new_raster_array,return_counts=True)