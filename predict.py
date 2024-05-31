import glob
import shutil
from collections import namedtuple
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data import ECGDataset
from models import EffNet
from training_models import BinaryClassificationModel
import matplotlib.pyplot as plt
import sklearn

code_path = Path('/Users/jonathanvo/Documents/ECG Single Lead ML/Code')
model_path = code_path / 'lightning_logs/version_57/checkpoints/epoch=45-step=276.ckpt'


project = code_path / "Dataset"
targets_file = project / "Manifest" / "DigitizedECGs_ExternalReplication_5_10_2022_removed.csv"
targets = pd.read_csv(targets_file)['case']

data_path = project / "test_output"

# weights = torch.load(model_path)


# Run = namedtuple("Run", ["name", "data_path", "manifest_path", "weights", "split"])
# run = [
#     Run(
#         "sudden_cardiac_death_final",
#         project / "test_output",
#         project / "Manifest/DigitizedECGs_ExternalReplication_5_10_2022_removed.csv",
#         weights,
#         None,
#     )
# ]
if __name__ == '__main__':
    test_ds = ECGDataset(
            data_path=data_path,
            manifest_path=targets_file,
            labels="case",
        )
    
    test_dl = DataLoader(
            test_ds, num_workers=0, batch_size=500, drop_last=False, shuffle=False
        )
    trainer = Trainer()
    classifier = BinaryClassificationModel()
    preds = trainer.predict(model=classifier, ckpt_path=model_path,dataloaders=test_dl)
    test_df = pd.DataFrame(preds)
    
    ecg_preds_test = np.concatenate((test_df[1][0], test_df[1][1], test_df[1][2]))
    ecg_name_test = test_df[0][0] + test_df[0][1] +test_df[0][2]
    
    print(f"AUC = {sklearn.metrics.roc_auc_score(targets,ecg_preds_test)}")
    fpr, tpr, threshold = sklearn.metrics.roc_curve(targets,ecg_preds_test)
    plt.plot(fpr,tpr)