# This script analyses the results obtained using unsupervised learning

import pandas as pd
import numpy as np
import useful_functions as my_functions
import matplotlib.pyplot as plt
import torch
from numpy import newaxis
from torchmetrics import JaccardIndex
from sklearn.metrics import confusion_matrix

PATH_local = 'C:/Users/agati/OneDrive - University College Dublin/Datasets/20200714 FrontierDevelopmentLab/LowResolution/'
PATH_COMPLEXITY = 'C:/Users/agati/Documents/DatasetsLocal/TopologicalAnalysis/'
PATH_RESULTS = '/results/'

tile_height = 20
tile_width = 20
training_data_size = 0.2
random_state_seed_number = 0

results = []

list_to_process = ['Mumbai','Capetown']

for location in list_to_process:

    # Reading ground-truth data
    file_ground_truth = location + '_ground_truth' + '.tif'
    path_with_gt_file = PATH_local+file_ground_truth
    image_array = my_functions.read_image_with_rasterio(path_with_gt_file)
    image_reduced = my_functions.reduce_image_for_irregular_tile_cutting(image_array=image_array,tile_height=tile_height,tile_width=tile_width)
    unique_values, _ = np.unique(image_reduced, return_counts=True)
    if len(unique_values) > 2:
        image_reduced_binary = np.where(image_reduced <= 1, image_reduced, 0)
    else:
        image_reduced_binary = image_reduced
    np.unique(image_reduced_binary,return_counts=True)

    # Reading the complexity raster
    file_complexity = location.lower() + '_complexity.tif'
    raw_complexity_raster = my_functions.read_image_with_rasterio(PATH_COMPLEXITY+file_complexity)
    complexity_reduced = my_functions.reduce_image_for_irregular_tile_cutting(image_array=raw_complexity_raster,tile_height=tile_height,tile_width=tile_width)
    np.unique(complexity_reduced,return_counts=True)

    # Filtering pixels that have a complexity value
    flat_ground_truth = image_reduced_binary.ravel()
    flat_complexity = complexity_reduced.ravel()
    have_complexity_info_array = np.where(flat_complexity < 99)
    filtered_complexity = flat_complexity[have_complexity_info_array]
    filtered_ground_truth = flat_ground_truth[have_complexity_info_array]

    # Analysing pixels that do not have a complexity value
    do_not_have_complexity_info_array = np.where(flat_complexity == 99)
    no_complexity = flat_complexity[do_not_have_complexity_info_array]
    no_complexity_ground_truth = flat_ground_truth[do_not_have_complexity_info_array]
    _, counts_no_complexity_gt = np.unique(no_complexity_ground_truth,return_counts=True)
    _, counts_gt = np.unique(flat_ground_truth,return_counts=True)
    total_pixels = flat_ground_truth.shape[0]
    total_no_complexity = sum(counts_no_complexity_gt)
    percentage_no_complexity = int(100*sum(counts_no_complexity_gt)/flat_ground_truth.shape[0])
    percentage_slums_no_complexity = int(100*counts_no_complexity_gt[1]/counts_gt[1])
    print('Of the {} pixels in the imagery of {}, {} percent do not have complexity info. Of the {} that do not have info, {} are nonslums and {} are slums.'.format(total_pixels,location,percentage_no_complexity,total_no_complexity,counts_no_complexity_gt[0],counts_no_complexity_gt[1]))
    print('Since the ground-truth has {} slum pixels, that means that {} ({} percentage) do not have complexity info.'.format(counts_gt[1],counts_no_complexity_gt[1],percentage_slums_no_complexity))

    # Plotting the complexity for each class (at pixel level)        
    nonslum_pixels_indices = np.where(flat_ground_truth == 0)
    nonslum_complexity = flat_complexity[nonslum_pixels_indices]
    x, y = np.unique(nonslum_complexity,return_counts=True)
    plt.title("Complexity of nonslum pixels ({})".format(location))
    plt.bar(x=x[:-1],height=y[:-1])
    plt.savefig('results_images/complexity_nonslum_auto_{}.png'.format(location))
    slum_pixels_indices = np.where(flat_ground_truth == 1)
    slum_complexity = flat_complexity[slum_pixels_indices]
    x_slum, y_slum = np.unique(slum_complexity,return_counts=True)
    plt.title("Complexity of slum pixels ({})".format(location))
    plt.bar(x=x_slum[:-1],height=y_slum[:-1])
    plt.savefig('results_images/complexity_slum_auto_{}.png'.format(location))

    # Calculating complexity for each tile in the location
    complexity_reduced_3_channels = complexity_reduced[:, :, newaxis]
    tiled_array = my_functions.reshape_split(complexity_reduced_3_channels, kernel_size=(tile_height,tile_width))
    tile_ID = []
    counter = -1
    can_calculate_tile_complexity = []
    average_complexity_tile = []
    total_pixels_in_tile = tile_height * tile_width
    for i in range(0,tiled_array.shape[0]):
        for j in range(0,tiled_array.shape[1]):
            counter = counter + 1
            tile_ID.append(counter)   
            my_tile = tiled_array[i,j,:,:,0]
            total_pixel_no_complexity = np.count_nonzero(my_tile == 99)
            if total_pixel_no_complexity >= 0.5 * tile_height * tile_width:      
                can_calculate_tile_complexity.append(0)
            else:
                can_calculate_tile_complexity.append(1)
            my_tile_filtered_99 = my_tile[my_tile < 99]
            if len(my_tile[my_tile < 99]) == 0:
                average_complexity_tile.append(99)
            else:
                average_complexity_tile.append(np.mean(my_tile_filtered_99))
 
    df_complexity = pd.DataFrame(np.column_stack([tile_ID, average_complexity_tile, can_calculate_tile_complexity]), 
                                columns=['tile_ID','average_complexity_tile','can_calculate_tile_complexity'])
    df_complexity[['tile_ID','can_calculate_tile_complexity']] = df_complexity[['tile_ID','can_calculate_tile_complexity']].astype('int32')

    # Reading baseline
    raw_baseline = pd.read_csv('results/df_results_baseline_CNN_{}.csv'.format(location))
    raw_baseline = raw_baseline.drop(columns=['Unnamed: 0'])

    # Reading cluster
    raw_cluster = pd.read_csv('results/df_features_test_{}.csv'.format(location))
    raw_cluster = raw_cluster.drop(columns=['Unnamed: 0'])
    
    # Merging the results
    check_test_tiles_are_the_same = list(raw_baseline['tile_ID'])==list(raw_cluster['tile_ID'])
    print('It is _{}_ that the tiles "tile_ID" is the same in the baseline and cluster for {}.'.format(check_test_tiles_are_the_same,location) )
    df_comparison = pd.merge(raw_baseline,raw_cluster,on="tile_ID", how="left")

    # Tiles in each cluster per class
    grouped_by_cluster = df_comparison[['all_tiles_class','cluster_ID']].groupby(['all_tiles_class','cluster_ID']).agg(Count=('cluster_ID','count'))
    grouped_by_cluster.to_csv('results/grouped_by_cluster_{}.csv'.format(location))

    # Median of each cluster
    df_comparison_complete = pd.merge(df_comparison,df_complexity,on="tile_ID", how="left")
    only_tiles_can_calculate_complexity = df_comparison_complete[df_comparison_complete['can_calculate_tile_complexity']==1]
    only_tiles_can_calculate_complexity = only_tiles_can_calculate_complexity[['cluster_ID','average_complexity_tile']]
    median_per_cluster = only_tiles_can_calculate_complexity.groupby(['cluster_ID']).agg(Median=('average_complexity_tile','median'))
    median_per_cluster.to_csv('results/median_per_cluster_{}.csv'.format(location))

    # Clusters with less than median complexity
    median_complexity_for_location = np.median(median_per_cluster['Median'])
    print('The median for {} is {}'.format(location,median_complexity_for_location))
    clusters_slum = []
    for i in range(0,len(median_per_cluster)):
        if median_per_cluster['Median'].iloc[i] < median_complexity_for_location:
            clusters_slum.append(median_per_cluster.index[i])
    print('The slum clusters for {} are {}'.format(location,clusters_slum))

    # Results unsupervised learning
    unsupervised_pred_labels = []
    for i in range(0,len(df_comparison_complete)):
        if int(df_comparison_complete['cluster_ID'].iloc[i]) in clusters_slum:
            unsupervised_pred_labels.append(int(1))
        else:
            unsupervised_pred_labels.append(int(0))
    df_comparison_complete = df_comparison_complete.assign(unsupervised_pred_labels=unsupervised_pred_labels)
    df_comparison_complete.to_csv('results/df_results_unsupervised_{}.csv'.format(location))    

    # Calculating metrics for unsupervised learning    
    iou = JaccardIndex(num_classes=2, reduction='none')
    IoU_nonslum_slum = iou(torch.tensor(df_comparison_complete['unsupervised_pred_labels']), torch.tensor(df_comparison_complete['all_tiles_class']))   
    iou = JaccardIndex(num_classes=2)
    meanIoU = iou(torch.tensor(df_comparison_complete['unsupervised_pred_labels']), torch.tensor(df_comparison_complete['all_tiles_class']))
    print('The results for a tile level classification considering slum any tile in a cluster with less than the median complexity for the locaiton:')
    print(IoU_nonslum_slum)
    print(meanIoU)

    confusion_matrix(df_comparison_complete['unsupervised_pred_labels'],df_comparison_complete['all_tiles_class'])
    ## iou = true_positives / (true_positives + false_positives + false_negatives)

    results.append([location,training_data_size,random_state_seed_number,float(IoU_nonslum_slum[0]),float(IoU_nonslum_slum[1]),float(meanIoU)])

df_results_final = pd.DataFrame(results, columns = ["Location", "Training data size", "Random Seed Number", "IoU nonSlum", "IoU Slum", "meanIoU"])

df_results_final.to_csv("results/df_results_unsupervised.csv")