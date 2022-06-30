# This script generates features (unsupervised learning) for each location

import numpy as np
import os
import pandas as pd
import rasterio
import joblib
import torch
import torch.backends.cudnn as cudnn
import useful_functions as my_functions
from torchvision import models, transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

PATH_local = 'C:/Users/agati/OneDrive - University College Dublin/Datasets/20200714 FrontierDevelopmentLab/LowResolution/'
PATH_server = 'comparison/'

try:
    list_of_files = os.listdir(PATH_server)
    myPATH = PATH_server
except FileNotFoundError:
    list_of_files = os.listdir(PATH_local)
    myPATH = PATH_local

# Defining useful functions
def step_one_part1(path_with_file,tile_height,tile_width):
    """Reads sattelite image, normalise it to 0-255 and split it into tiles.

    Args:
        path_with_file (string): File path.
        tile_height (int): tile height.
        tile_width (int): tile width.

    Returns:
        array: Tiled imagery.
    """    
    with rasterio.open(path_with_file, 'r') as image:
        bandb1_array = image.read(1)
        bandb2_array = image.read(2)
        bandb3_array = image.read(3)
        print('The image has dtypes {}'.format(image.dtypes))
    band1_reduced = my_functions.reduce_image_for_irregular_tile_cutting(bandb1_array,tile_height=tile_height,tile_width=tile_width)
    band2_reduced = my_functions.reduce_image_for_irregular_tile_cutting(bandb2_array,tile_height=tile_height,tile_width=tile_width)
    band3_reduced = my_functions.reduce_image_for_irregular_tile_cutting(bandb3_array,tile_height=tile_height,tile_width=tile_width)
    # Scaling the data to convert to np.uint8
    transformer = MinMaxScaler().fit(band1_reduced)
    band1_int8 = transformer.transform(band1_reduced)*255
    band1_int8 = band1_int8.astype(np.uint8)
    transformer = MinMaxScaler().fit(band2_reduced)
    band2_int8 = transformer.transform(band2_reduced)*255
    band2_int8 = band2_int8.astype(np.uint8)
    transformer = MinMaxScaler().fit(band3_reduced)
    band3_int8 = transformer.transform(band3_reduced)*255
    band3_int8 = band3_int8.astype(np.uint8)
    all_bands = np.stack([band1_int8,band2_int8,band3_int8], axis=2)
    # Cutting the imagery into tiles
    all_bands_tiles = my_functions.reshape_split(all_bands,kernel_size=(tile_height,tile_width))
    return all_bands_tiles

def step_one_part2(path_with_gt_file,tile_height,tile_width, threshold=0.5):
    """Generates a dataframe with information about each tile and also an array with the class of each tile.

    Args:
        path_with_gt_file (str): File path.
        tile_height (int): tile height.
        tile_width (int): tile width.
        threshold (float, optional): Percentage of pixels labelled 1 for the whole tile to be considered 1. Defaults to 0.5.

    Returns:
        df: DataFrame with information.
        array: Array with the label of each tile.
    """    
    raw_file_gt = rasterio.open(path_with_gt_file)
    print('The ground_truth image has dtypes {}'.format(raw_file_gt.dtypes))
    image_array = my_functions.read_image_with_rasterio(path_with_gt_file)
    image_reduced = my_functions.reduce_image_for_irregular_tile_cutting(image_array=image_array,tile_height=tile_height,tile_width=tile_width)
    unique_values, _ = np.unique(image_reduced, return_counts=True)
    if len(unique_values) > 2:
        image_reduced_binary = np.where(image_reduced <= 1, image_reduced, 0)
    else:
        image_reduced_binary = image_reduced
    df, tile_code_array = my_functions.generating_df_and_code_array(image_array=image_reduced_binary,tile_height=tile_height,tile_width=tile_width,threshold=threshold)
    return df, tile_code_array

def step_two(df,training_data_size,multiplier_nonslum_slum,random_state_seed_number=0,number_of_channels=3):
    """Given a df and size of training data, returns X and y for training, validation, and test data.

    Args:
        df (dataframe): Dataframe with info about each tile
        training_data_size (float): Between 0.0 and 1.0
        random_state_seed_number (int, optional): Random state number. Defaults to 0.
        number_of_channels (int, optional): Number of channels in the imagery. Defaults to 3.

    Returns:
        dataframes: X_train, y_train, X_val, y_val, X_test, y_test, df_my_current_fold, df_my_current_fold_test
    """    
    number_of_tiles_training = int(training_data_size*sum(df['all_tiles_class']==1))
    X_train_nonslum = df[df['all_tiles_class']==0].sample(n=number_of_tiles_training*multiplier_nonslum_slum, random_state=random_state_seed_number)
    X_train_slum = df[df['all_tiles_class']==1].sample(n=number_of_tiles_training, random_state=random_state_seed_number)
    df_my_current_fold = pd.concat([X_train_nonslum, X_train_slum], axis = 0)
    X_y_train, X_y_val = train_test_split(df_my_current_fold, test_size=0.3, stratify=df_my_current_fold['all_tiles_class'], random_state=0)
    X_train = tiles_array[X_y_train['code_image_i'],X_y_train['code_image_j'],:,:,0:number_of_channels]
    y_train = np.array(X_y_train['all_tiles_class'])
    X_val = tiles_array[X_y_val['code_image_i'],X_y_val['code_image_j'],:,:,0:number_of_channels]
    y_val = np.array(X_y_val['all_tiles_class'])
    df_my_current_fold_test = df[~df.tile_ID.isin(df_my_current_fold['tile_ID'])]
    X_test = tiles_array[df_my_current_fold_test['code_image_i'],df_my_current_fold_test['code_image_j'],:,:,0:number_of_channels]
    y_test = np.array(df_my_current_fold_test['all_tiles_class'])
    return X_train, y_train, X_val, y_val, X_test, y_test, df_my_current_fold, df_my_current_fold_test

# Defining variables for processing
tile_height = 20
tile_width = 20
number_of_channels = 3    
threshold = 0.5
multiplier_nonslum_slum = 4
training_data_size = 0.2
random_state_seed_number = 0
list_to_process = ['Mumbai','Capetown']
cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for location in list_to_process:

    file = location + '.tif'
    file_ground_truth = location + '_ground_truth' + '.tif'
    path_with_file = myPATH+file
    path_with_gt_file = myPATH+file_ground_truth

    tiles_array = step_one_part1(path_with_file=path_with_file,tile_height=tile_height,tile_width=tile_width)
    df, _ = step_one_part2(path_with_gt_file=path_with_gt_file,tile_height=tile_height,tile_width=tile_width, threshold=threshold)

    slum_tiles = len(df[df['all_tiles_class']==1])
    nonslum_tiles = len(df[df['all_tiles_class']==0])

    print('The {} dataset has {} slum tiles and {} nonslum tiles.'.format(location,slum_tiles,nonslum_tiles))

    # Generating features
    model = models.resnet18(pretrained=True)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    array_output = np.zeros((len(df), 1000))
    k = 0
    for i in range(0,len(df)):
        input_tensor_my = preprocess(tiles_array[df['code_image_i'].iloc[i],df['code_image_j'].iloc[i],:,:,:])
        input_batch_my = input_tensor_my.unsqueeze(0)
        prob = model(input_batch_my)
        with torch.no_grad():
            array_output[k] = prob
        k = k + 1

    df_output = pd.DataFrame(array_output)
    df_output = df_output.assign(tile_ID=df['tile_ID'])
    df_output.to_csv("results/df_features_unsupervised_1000_{}.csv".format(location))

    # Preparing train and test sets
    _, _, _, _, _, _, df_my_current_fold, df_my_current_fold_test = step_two(df,training_data_size=training_data_size,multiplier_nonslum_slum=multiplier_nonslum_slum,random_state_seed_number=random_state_seed_number,number_of_channels=number_of_channels)

    X_train = df_output[df_output.tile_ID.isin(df_my_current_fold['tile_ID'])]
    X_train_array = np.array(X_train.drop(columns=['tile_ID']))

    X_test = df_output[df_output.tile_ID.isin(df_my_current_fold_test['tile_ID'])]
    X_test_array = np.array(X_test.drop(columns=['tile_ID']))

    # Checking baseline was trained with the same tiles used for training
    df_train = pd.read_csv('results/df_train_{}.csv'.format(location)) 
    check_train_tiles_are_the_same = list(df_my_current_fold['tile_ID'])==list(df_train['tile_ID'])
    print('It is _{}_ that the tiles used to train the baseline model are the same tiles being used to train the unsupervised model for {}.'.format(check_train_tiles_are_the_same,location) )
    df_test = pd.read_csv('results/df_test_{}.csv'.format(location)) 
    check_test_tiles_are_the_same = list(df_my_current_fold_test['tile_ID'])==list(df_test['tile_ID'])
    print('It is _{}_ that the tiles used to test the baseline model are the same tiles being used to train the unsupervised model for {}.'.format(check_test_tiles_are_the_same,location) )

    # Training the model
    model_kmeans = KMeans(n_clusters=17,init='k-means++',n_init=100, random_state=0)
    model_kmeans.fit(X_train_array)
    joblib.dump(model_kmeans, 'model_kmeans_{}.joblib'.format(location))

    # Assigning a cluster number to each tile
    yhat_test = model_kmeans.predict(X_test_array)
    df_test_unsupervised = pd.DataFrame(X_test['tile_ID'])
    df_test_unsupervised = df_test_unsupervised.assign(cluster_ID=yhat_test)
    yhat = model_kmeans.predict(array_output)
    df_all_dataset_unsupervised = pd.DataFrame(df_output['tile_ID'])
    df_all_dataset_unsupervised = df_all_dataset_unsupervised.assign(cluster_ID=yhat)

    # Saving results   
    df_test_unsupervised.to_csv("results/df_features_test_{}.csv".format(location))
    df_all_dataset_unsupervised.to_csv("results/df_features_all_dataset_{}.csv".format(location))