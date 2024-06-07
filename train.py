from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data import ECGDataset
from models import EffNet
import pandas as pd
import wandb
from training_models import BinaryClassificationModel

wandb.login(key='4fc049b731faf66e88c08e159d32d36d3ce58b1e')
run = wandb.init()

to_chkpt = '/Users/jonathanvo/Documents/ECG Single Lead ML/Code/lightning_logs'

def get_model(weights_path=None,chkpt_path=None):
    backbone = EffNet(output_neurons=1)

    Classifier = BinaryClassificationModel(backbone, early_stop_epochs=10, lr=0.001)

    if weights_path is not None:
        print(Classifier.load_state_dict(torch.load(weights_path)))
    
    if chkpt_path is not None:
        Classifier = BinaryClassificationModel(
            model=backbone,
            early_stop_epochs=10,
            lr=0.001
        )
        print(Classifier)

    return Classifier
    
def train(
    data_path,
    manifest_path,
    # train_manifest_path,
    # val_manifest_path,
    batch_size,
    num_workers,
    accelerator,
    max_epochs=100,
    weights_path=None,
    chkpt_path= None
):
    torch.cuda.empty_cache()
    
    full_ds = ECGDataset(
        data_path=data_path,
        labels="case",
        manifest_path=manifest_path,
        first_lead_only=False)
    
    train_size = int(0.8*len(full_ds))
    val_size = int((len(full_ds) - train_size)/2)
    test_size = val_size
    # train_ds,val_ds = torch.utils.data.random_split(full_ds,[train_size,val_size])
    train_ds,val_ds,test_ds = torch.utils.data.random_split(full_ds,[train_size,val_size,test_size])
    
    # train_ds = ECGDataset(
    #     data_path=data_path,
    #     labels="case",
    #     manifest_path=train_manifest_path,
    #     first_lead_only=False
    # )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )

    # val_ds = ECGDataset(
    #     data_path=data_path,
    #     labels="case",
    #     manifest_path=val_manifest_path,
    #     first_lead_only=False
    # )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )
    
    test_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        )

    model = get_model(weights_path,chkpt_path)

#CPU or GPU, can adjust here at in train function arguments, ddp vs dpp_notebook
    #trainer = Trainer(accelerator, max_epochs=max_epochs, strategy="ddp_notebook")
    trainer = Trainer(accelerator, max_epochs=max_epochs)
    trainer.fit(
        model, 
        train_dataloaders=train_dl, 
        val_dataloaders=val_dl,
        ckpt_path= chkpt_path)

project = Path('/Users/jonathanvo/Documents/ECG Single Lead ML/Code')
data_path = project / 'Dataset/output'
manifest_path = project / 'Dataset/Manifest/DigitizedECGs_4_13_2022_removed.csv'

#Read manifest spreadsheet, split into train and val, and write to two new spreadsheets
# train_set, val = train_test_split(pd.read_csv(manifest_path), test_size=0.2)
# train_set.to_csv(project / 'train_csv.csv')
# val.to_csv(project / 'val_csv.csv')
# manifest_train = project / 'train_csv.csv'
# manifest_val = project / 'val_csv.csv'

if __name__ == "__main__":
    args = dict(
        max_epochs=1000,
        accelerator="gpu",
        num_workers=0,
        batch_size=500,
        data_path=data_path,
        manifest_path=manifest_path,
        # train_manifest_path=manifest_train,
        # val_manifest_path=manifest_val,
        weights_path=None,
        chkpt_path= None)

    train(**args)

