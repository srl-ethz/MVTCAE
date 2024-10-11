import torch
from modalities.Modality import Modality

class RobotAction(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name, action_dim):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
        self.data_size = torch.Size([action_dim])
        self.gen_quality_eval = False
        self.file_suffix = '.npy'

    def save_data(self, d, fn, args):
        torch.save(d, fn)

    def plot_data(self, d):
        # For now, just return the data as is
        return d