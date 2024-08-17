# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch
from datetime import datetime

class MyCustomDataset(Dataset):
    # Define your custom dataset
    def __init__(self):
        # Initialize dataset, load data, etc.
        
        x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk, \
        y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk = dataImport()
        
        self.data   =   torch.cat((x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk), dim=0)
        self.labels = torch.cat((y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk), dim=0)
        print(self.data.shape)
        print(self.labels.shape)
        # print(self.data.shape,type(self.data))

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve a sample
        return self.data[idx], self.labels[idx]  

def txt_import():
    # Load data from csv files
    x_dic = {}
    y_dic = {}
    print("Importing txt files...")

    beg_time = datetime.now()
    for i in ["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]:
        
        label = str(i)
        start_time = datetime.now()
  
        xx_txt = "../../Dataset/xx_1000_60_txt" + label + ".csv"
        yy_txt = "../../Dataset/yy_1000_60_txt" + label + ".csv"
        
        arrXX = np.loadtxt(xx_txt,  delimiter=',', dtype=np.float32) 
        arrYY = np.loadtxt(yy_txt,  delimiter=',', dtype=np.int8) 
        arrXX = arrXX.reshape(-1, 500, 90)
        time_interval = datetime.now() - start_time
        print(label, "\t", time_interval.seconds, "seconds", "\t xx.shape:", np.shape(arrXX), "\t yy.shape", np.shape(arrYY))
        x_dic[label] = arrXX
        y_dic[label] = arrYY
    
    time_interval = datetime.now() - beg_time
    print("\nTotal time for txt_import: ", time_interval.seconds, "seconds")
    return x_dic["bed"], x_dic["fall"], x_dic["pickup"], x_dic["run"], x_dic["sitdown"], x_dic["standup"], x_dic["walk"], \
        y_dic["bed"], y_dic["fall"], y_dic["pickup"], y_dic["run"], y_dic["sitdown"], y_dic["standup"], y_dic["walk"]

def dataImport():
    # Import data
  
    x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk, \
    y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk = txt_import()              
    print("bed =", len(x_bed), " fall=", len(x_fall), " pickup =", len(x_pickup), " run=", len(x_run), " sitdown=", len(x_sitdown), " standup =", len(x_standup), " walk =", len(x_walk))
    
    x_bed = torch.from_numpy(x_bed)
    x_fall = torch.from_numpy(x_fall)
    x_pickup = torch.from_numpy(x_pickup)
    x_run = torch.from_numpy(x_run)
    x_sitdown = torch.from_numpy(x_sitdown)
    x_standup = torch.from_numpy(x_standup)
    x_walk = torch.from_numpy(x_walk)
    
    y_bed  = torch.from_numpy(y_bed)
    y_fall = torch.from_numpy(y_fall)
    y_pickup = torch.from_numpy(y_pickup)
    y_run = torch.from_numpy(y_run)
    y_sitdown = torch.from_numpy(y_sitdown)
    y_standup = torch.from_numpy(y_standup)
    y_walk = torch.from_numpy(y_walk)
    
    return  x_bed, x_fall, x_pickup, x_run, x_sitdown, x_standup, x_walk, \
    y_bed, y_fall, y_pickup, y_run, y_sitdown, y_standup, y_walk 

def getData():  
    # Split the original dataset into K subsets. For each subset i, use it as the validation set and the remaining K-1 subsets as the training set.
    kk = 5
    beg_time = datetime.now()
    print("\ngetData start")
    # Instantiate the dataset
    dataset = MyCustomDataset()
    # Define the size ratio of the training set and the test set, for example, 70% training set, 30% test set
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    total_time = datetime.now() - beg_time
    print("\ngetData time taken ", total_time.seconds, "seconds")
 
    return train_dataset, test_dataset
