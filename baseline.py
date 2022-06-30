# This script trains a supervised model for each location

import numpy as np
import os
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import useful_functions as my_functions
from torchmetrics import JaccardIndex
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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

def step_three(X_train, y_train, X_val, y_val, X_test, y_test, batch_size = 4):
    """Generates dataloaders that can be used to train a Pytorch model.

    Args:
        X_train (array): Numpy array with train data
        y_train (array): Numpy array with train labels
        X_val (array): Numpy array with validation data
        y_val (array): Numpy array with validation labels
        X_test (array): Numpy array with test data
        y_test (array): Numpy array with test labels

    Returns:
        DataLoader: Dataloaders that can be used in Pytorch model.
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip()
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = my_functions.MyDataset(data=X_train, targets = y_train, transform=preprocess)
    val_dataset = my_functions.MyDataset(data=X_val, targets = y_val, transform=preprocess)
    test_dataset = my_functions.MyDataset(data=X_test, targets = y_test, transform=preprocess)
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader

# Defining variables for processing
tile_height = 20
tile_width = 20
number_of_channels = 3    
threshold = 0.5
multiplier_nonslum_slum = 4
batch_size = 8
training_data_size = 0.2
random_state_seed_number = 0
list_to_process = ['Mumbai','Capetown']

results = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

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

    X_train, y_train, X_val, y_val, X_test, y_test, df_my_current_fold, df_my_current_fold_test = step_two(df,training_data_size=training_data_size,multiplier_nonslum_slum=multiplier_nonslum_slum,random_state_seed_number=random_state_seed_number,number_of_channels=number_of_channels)   
    df_my_current_fold.to_csv('results/df_train_{}.csv'.format(location))
    df_my_current_fold_test.to_csv('results/df_test_{}.csv'.format(location))

    train_loader, val_loader, test_loader = step_three(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size)

    # Checking tiles of each category in the train and test sets
    tiles_slum_train = int(sum(y_train == 1))
    tiles_nonslum_train = int(sum(y_train == 0))
    tiles_training_set = len(y_train)
    tiles_slum_test = int(sum(y_test == 1))
    tiles_nonslum_test = int(sum(y_test == 0))

    print('The train set contains {} tiles ({} slums and {} nonslums).'.format(tiles_training_set,tiles_slum_train,tiles_nonslum_train))
    print('The test set contains {} tiles ({} slums and {} nonslums).'.format((tiles_slum_test+tiles_nonslum_test),tiles_slum_test,tiles_nonslum_test))

    # Setting the model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters())

    my_trained_model = my_functions.my_train_model(train_loader,val_loader,model_ft,criterion,optimizer_ft,num_epochs=50,patience=10)
    torch.save(my_trained_model.state_dict(), 'models/baseline_CNN_{}_repeat.pth'.format(location))
    
    # Loading the model
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)
    # model_ft = model_ft.to(device)
    # model_ft.load_state_dict(torch.load("models/baseline_CNN_{}.pth".format(location)))
    # model_ft.eval()
    # my_trained_model = model_ft

    # Testing the model on the testing loader
    pred_labels_list = []
    for test_batch, (X, y) in enumerate(test_loader):
        images, labels = X, y
        outputs = my_trained_model(images)
        _, predicted = torch.max(outputs, 1)
        pred_labels_list = pred_labels_list + predicted.tolist()
    df_results = df_my_current_fold_test[['tile_ID','all_tiles_class']]
    df_results = df_results.assign(pred_labels=pred_labels_list)
    df_results.to_csv('results/df_results_baseline_CNN_{}.csv'.format(location))

    # Calculating IoU
    iou = JaccardIndex(num_classes=2, reduction = 'none')
    IoU_nonslum_slum = iou(torch.tensor(np.array(df_results['pred_labels'])), torch.tensor(np.array(df_results['all_tiles_class'])))
    iou = JaccardIndex(num_classes=2)
    meanIoU = iou(torch.tensor(np.array(df_results['pred_labels'])), torch.tensor(np.array(df_results['all_tiles_class'])))
    print(IoU_nonslum_slum)
    print(meanIoU)

    # Appending results
    results.append([location,training_data_size,random_state_seed_number,float(IoU_nonslum_slum[0]),float(IoU_nonslum_slum[1]),float(meanIoU)])

df_results_final = pd.DataFrame(results, columns = ["Location", "Training data size", "Random Seed Number", "IoU Slum", "IoU nonSlum", "meanIoU"])

df_results_final.to_csv("results/df_results_baseline_CNN_two_locations.csv")