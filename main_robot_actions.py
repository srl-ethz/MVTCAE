import sys
import os
import json

import torch

from run_epochs import run_epochs
from utils.filehandling import create_dir_structure
from robot_actions.flags import parser
from robot_actions.experiment import RobotActionsExperiment

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

    FLAGS.class_dim = 64
    FLAGS.style_dim = 32
    FLAGS.likelihood = 'normal'

    # postprocess flags

    if FLAGS.div_weight_uniform_content is None:
        FLAGS.div_weight_uniform_content = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content]
    if FLAGS.div_weight is None:
        FLAGS.div_weight = 1 / (FLAGS.num_mods + 1)
    FLAGS.alpha_modalities.extend([FLAGS.div_weight for _ in range(FLAGS.num_mods)])
    print("alpha_modalities:", FLAGS.alpha_modalities)

    create_dir_structure(FLAGS)

    exp = RobotActionsExperiment(FLAGS)
    exp.set_optimizer()

    print(FLAGS)
    run_epochs(exp)