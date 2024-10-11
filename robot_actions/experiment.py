import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from robot_actions.RobotActionsDataset import RobotActionsDataset

from utils.BaseExperiment import BaseExperiment
from modalities.RobotAction import RobotAction
from robot_actions.networks.RobotActionNetworks import EncoderAction, DecoderAction
from robot_actions.networks.VAERobotActions import VAERobotActions

class RobotActionsExperiment(BaseExperiment):
    def __init__(self, flags):
        super().__init__(flags)
        self.num_modalities = flags.num_mods
        self.modalities = self.set_modalities()
        self.subsets = self.set_subsets()
        self.mm_vae = self.set_model()
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.dataset_train = None
        self.dataset_test = None
        self.set_dataset()
        
        self.clfs = self.set_clfs()
        self.test_samples = self.get_test_samples()
        self.eval_metric = self.mean_eval_metric
        self.paths_fid = self.set_paths_fid()

        self.labels = ['action_class']

    def set_model(self):
        model = VAERobotActions(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device)
        return model

    def set_modalities(self):
        mods = [
            RobotAction("faive_angles", 
                        EncoderAction(self.flags, input_dim=11),
                        DecoderAction(self.flags, output_dim=11),
                        self.flags.class_dim, self.flags.style_dim, self.flags.likelihood, 11),
            RobotAction("mano_pose", 
                        EncoderAction(self.flags, input_dim=45),
                        DecoderAction(self.flags, output_dim=45),
                        self.flags.class_dim, self.flags.style_dim, self.flags.likelihood, 45),
            RobotAction("simple_gripper", 
                        EncoderAction(self.flags, input_dim=1),
                        DecoderAction(self.flags, output_dim=1),
                        self.flags.class_dim, self.flags.style_dim, self.flags.likelihood, 1)
        ]
        return {m.name: m for m in mods}

    def set_dataset(self):
        full_dataset = RobotActionsDataset(self.flags.data_path)
        dataset_size = len(full_dataset)
        train_size = int(dataset_size * self.flags.train_ratio)
        test_size = dataset_size - train_size
        
        self.dataset_train, self.dataset_test = torch.utils.data.random_split(
            full_dataset, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

    def set_optimizer(self):
        params = list(self.mm_vae.parameters())
        self.optimizer = optim.Adam(params, 
                                    lr=self.flags.initial_learning_rate,
                                    betas=(self.flags.beta_1, self.flags.beta_2))

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = 1.0
        return rec_weights

    def set_style_weights(self):
        return {m: self.flags.beta_style for m in self.modalities.keys()}
    
    def get_test_samples(self, num_samples=10):
        indices = torch.randperm(len(self.dataset_test))[:num_samples]
        samples = [self.dataset_test[i] for i in indices]
        return [{k: v.to(self.flags.device) for k, v in sample.items()} for sample in samples]

    def set_clfs(self):
        clfs = {m: None for m in self.modalities.keys()}
        
        if self.flags.use_clf:
            raise NotImplementedError("Classifiers are not implemented for RobotActionsExperiment")
        
        return clfs

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def get_prediction_from_attr(self, attr, index=None):
        return np.argmax(attr, axis=1).astype(int)

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)

    # def set_paths_fid(self):
    #     # Implement FID paths setup if needed
    #     return None