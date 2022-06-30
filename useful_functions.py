# This script contains functions that are used multiple times in different scripts

import rasterio
import numpy as np
import pandas as pd
import torch
import time
import copy
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class MyDataset(Dataset):
    """Creates a Pytorch dataset from numpy objects (data and targets) and applies a transformation to the data.
    Modified from https://stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader

    Args:
        data (numpy.ndarray): each image in the training set in the format (width, height, channels)
        targets (numpy.ndarray): vector with labels
        transform (torchvision.transforms): compose of the transformations to be applied to data

    Returns:
        DataSet: a DataSet in a Pytorch format
    """
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            # x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)

def my_train_model(train_loader,val_loader, model, loss_fn, optimizer, num_epochs=25, patience=5):
    """Trains a model using Pytorch.
    Adapted from https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    Args:
        train_loader (DataLoader): training set.
        val_loader (DataLoader): validation set.
        model (Model): model to be fine-tuned.
        loss_fn (Torch loss): how the loss will be calculated. Example: cross entropy.
        optimizer (Torch optimizer): optimizer. Example: Adam.
        num_epochs (int, optional): Number of Epochs. Defaults to 25.
        patience (int, optional): Number of epochs to evaluate to decide to early stop the training. Early stopping is activated when validation loss is more than 30% of the loss in the previous patience-number epochs. Defaults to 5 epochs.

    Returns:
        model: Pytorch model with the best accuracy in the validation set.
    """

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    val_loss_list = []    

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        train_size = len(train_loader.dataset)

        # Training phase
        for train_batch, (X, y) in enumerate(train_loader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batch % 100 == 0:
            loss, current = loss.item(), train_batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
        
        # Validation phase
        val_size = len(val_loader.dataset)
        num_batches = len(val_loader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for X, y in val_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= val_size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_wts = copy.deepcopy(model.state_dict())    

        val_loss_list.append(test_loss)
        if epoch > patience:
            if test_loss >= (np.mean(val_loss_list[-patience:-1]))*1.2:
                print('Training was early stopped because current validation loss is more than 20 percent of the average of the last %d epochs.' %patience)
                break
        
        # After train and validation phases, scheduler operates
        # scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def read_image_with_rasterio(path):
    """Reads raster image (.tif file) and saves the first layer of the raster as a numpy array. Converts the array to int8.
    Args:
        path (tif): Raster image

    Returns:
        array: array with the imagery
    """
    with rasterio.open(path, 'r') as image:
        array = image.read(1)
        # array = array.astype(np.uint8)
    return array

def read_image_with_rasterio_all_bands_and_reduce_it_for_tiling(path_with_file, tile_height,tile_width):
    """Reads raster image (.tif file) and saves the first three layers of the raster as a numpy array. Converts the array to int8.
    Args:
        path (tif): Raster image (channels, height, width)

    Returns:
        array: array with the imagery
    """
    with rasterio.open(path_with_file, 'r') as image:
        array = image.read()#.astype(np.uint8)
        number_of_tiles_i = int(array.shape[1]/tile_height)
        new_number_rows = number_of_tiles_i*tile_height
        number_of_tiles_j = int(array.shape[2]/tile_width)    
        new_number_columns = number_of_tiles_j*tile_width
        image_reduced = array[:,0:new_number_rows, 0:new_number_columns]
    return image_reduced

def reduce_image_for_irregular_tile_cutting(image_array,tile_height,tile_width):
    """Adjusts the number of rows and columns so the image can be cut into tiles of the desired dimensions.

    Args:
        image_array (numpy.ndarray): array with original imagery.
        tile_height (int): height of each tile.
        tile_width (int): width of each tile.

    Returns:
        array: Image reduced
    """
    number_of_tiles_i = int(image_array.shape[0]/tile_height)
    new_number_rows = number_of_tiles_i*tile_height
    number_of_tiles_j = int(image_array.shape[1]/tile_width)    
    new_number_columns = number_of_tiles_j*tile_width
    image_reduced = image_array[0:new_number_rows, 0:new_number_columns]
    return image_reduced

def reshape_split(image, kernel_size):
    """Reshapes an array into tiles the a desired kernel size. 
    Developed by: Iosif Doundoulakis. 
    Source: https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7

    Args:
        image (array): Array to be resized in the format (img_height, img_width, channels).
        kernel_size (tuple): A tuple with (tile_height, tile_width) of the desired kernel.

    Returns:
        array: Reshaped array.
    """
    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape(img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels)
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

def generating_df_and_code_array(image_array, tile_height, tile_width, threshold):
    """Generates a dataframe with information about each tile and also an array with the class of each tile.

    Args:
        image_array (array): Array that will be used to generate dataframe.
        tile_height (int): height of each tile.
        tile_width (int): width of each tile.
        threshold (float): Percentage of pixels labelled 1 for the whole tile to be considered 1.

    Returns:
        df: DataFrame with information.
        array: Array with the label of each tile.
    """

    # Creating tile code array
    image_reduced_reshaped = np.reshape(image_array,(image_array.shape[0], image_array.shape[1],1))
    tiled_array = reshape_split(image_reduced_reshaped, kernel_size=(tile_height,tile_width))
    tile_code = np.zeros((tiled_array.shape[0],tiled_array.shape[1]))
    code_i_tile_slums = []
    code_j_tile_slums = []
    percentage_tiles_slum = []
    number_of_pixels_slum_per_tile = []
    for i in range(0,tiled_array.shape[0]):
        for j in range(0,tiled_array.shape[1]):
            number_of_pixels_slum = sum(sum(tiled_array[i,j,:,:,0]==1))
            number_of_pixels_slum_per_tile.append(number_of_pixels_slum)
            proportion_pixels_slum_in_tile = number_of_pixels_slum/(tile_height*tile_width)
            percentage_tiles_slum.append(proportion_pixels_slum_in_tile)
            if number_of_pixels_slum >= threshold*tile_height*tile_width:      
                tile_code[i,j]=1
                code_i_tile_slums.append(i)
                code_j_tile_slums.append(j)
            else:
                tile_code[i,j]=0

    # Generating the df 
    all_tiles_class = []
    code_image_i = []
    code_image_j = []
    tile_ID = []
    counter = -1
    for i in range(0,tile_code.shape[0]):
        for j in range(0,tile_code.shape[1]):
            counter = counter + 1
            all_tiles_class.append(tile_code[i,j])
            code_image_i.append(i)
            code_image_j.append(j)
            tile_ID.append(counter)      

    df = pd.DataFrame(np.column_stack([tile_ID, code_image_i, code_image_j, all_tiles_class,number_of_pixels_slum_per_tile,percentage_tiles_slum]), 
                                columns=['tile_ID','code_image_i', 'code_image_j', 'all_tiles_class','number_of_pixels_slum_per_tile','percentage_tiles_slum'])
    df[['tile_ID','code_image_i','code_image_j','all_tiles_class']] = df[['tile_ID','code_image_i','code_image_j','all_tiles_class']].astype('int32')

    return df, tiled_array

def generate_tile_array_and_df_and_code_array(path_with_file,path_with_gt_file,tile_height,tile_width,threshold):
    """Generates tiles_array from .tif imagery. Generates df with info about tiles given a tile dimension.

    Args:
        path_with_file (string): Path to file with imagery
        file_ground_truth (string): Path with ground truth data
        tile_height (int): Tile height
        tile_width (int): Tile width
        threshold (float): Percentage of pixels labelled 1 for the whole tile to be considered 1.


    Returns:
        tiles_array (array): imagery in a tile format
        df (dataframe): Dataframe with info about each tile
        tile_code_array (array): tile code array

    """
    # Processing imagery data
    resized_array = read_image_with_rasterio_all_bands_and_reduce_it_for_tiling(path_with_file=path_with_file,tile_height=tile_height,tile_width=tile_width)
    resized_array_channels_last = np.moveaxis(resized_array,0,2)
    tiles_array = reshape_split(resized_array_channels_last,(tile_height,tile_width))
    # Processing ground truth
    image_array = read_image_with_rasterio(path_with_gt_file)
    image_reduced = reduce_image_for_irregular_tile_cutting(image_array=image_array,tile_height=tile_height,tile_width=tile_width)
    unique_values, _ = np.unique(image_reduced, return_counts=True)
    if len(unique_values) > 2:
        image_reduced_binary = np.where(image_reduced <= 1, image_reduced, 0)
    else:
        image_reduced_binary = image_reduced
    df, tile_code_array = generating_df_and_code_array(image_array=image_reduced_binary,tile_height=tile_height,tile_width=tile_width,threshold=threshold)
    return tiles_array, df, tile_code_array

def generate_datasets_training(df,tiles_array,training_data_size,random_state_seed_number=0,number_of_channels=3):
    """Given a df and size of training data, returns X and y for training, validation and test data.

    Args:
        df (dataframe): Dataframe with info about each tile.
        tiles_array (array): Array with all the data (tiled).
        training_data_size (float): Between 0.0 and 1.0.
        random_state_seed_number (int, optional): Random state number. Defaults to 0.
        number_of_channels (int, optional): Number of channels in the imagery. Defaults to 3.

    Returns:
        dataframes: X_train, y_train, X_val, y_val, X_test, y_test, df_my_current_fold, df_my_current_fold_test
    """    
    number_of_tiles_training = int(training_data_size*sum(df['all_tiles_class']==1))
    X_train_nonslum = df[df['all_tiles_class']==0].sample(n=number_of_tiles_training, random_state=random_state_seed_number)
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