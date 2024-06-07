from pathlib import Path
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    #Loads ECG data files (names, values) into python
    #Data_path is the path of the actual waveform data
    #Manifest_path is the path of excel spreadsheet containing values and filenames
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        first_lead_only=True,
    ):
        self.data_path = Path(data_path) #actual waveform path
        self.split = split

        self.labels = labels
        if (self.labels is not None) and isinstance(self.labels, str):
            self.labels = [self.labels]

        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        self.manifest = pd.read_csv(self.manifest_path, low_memory=False) #Reading CSV of ECG names

        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]

        self.filenames_list = list(self.manifest["FileName"]) #List of ECG names

        if self.labels is not None:
            self.labels_array = self.manifest[self.labels].to_numpy() #1Binary SCD values

        self.first_lead_only = first_lead_only

    def read_file(self, filepath):
        #Function reads waveform file y-values converted by convert_AMPS_to_npy script
        file = np.load(filepath)
        if file.shape[0] != 12:
            file = file.T
        file = torch.tensor(file).float()

        if self.first_lead_only:
            file = file[0:1]
        
        return file
    
    def plt_ecg(self,filepath,lead_number):
        file = np.load(filepath)
        if file.shape[0] != 12:
            file = file.T
        file = torch.tensor(file).float()
        file = file[lead_number]
        x_values = np.arange(0,len(file),1)
        plt.plot(x_values, file)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        filename = self.filenames_list[index] #Finding filename based on index from the name list
        if self.labels is not None:
            y = self.labels_array[index]
        else:
            y = None

        filepath = self.data_path / (filename + ".npy") #Setting filepath of the actual .npy waveform
        x = self.read_file(filepath)

        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if self.labels is not None and not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)
            return filename, x, y
        else:
            return filename, x
