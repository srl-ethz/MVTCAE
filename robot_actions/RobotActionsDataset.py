import numpy as np
import torch
from torch.utils.data import Dataset

class RobotActionsDataset(Dataset):
    def __init__(self, file_path):
        self.data = np.load(file_path, allow_pickle=True).item()  # Load as dictionary
        self.length = len(self.data['faive_angles'])  # Assume all arrays have the same length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            'faive_angles': torch.tensor(self.data['faive_angles'][idx], dtype=torch.float32),
            'mano_pose': torch.tensor(self.data['pose'][idx], dtype=torch.float32),
            'simple_gripper': torch.tensor(self.data['simple_gripper'][idx], dtype=torch.float32)
        }