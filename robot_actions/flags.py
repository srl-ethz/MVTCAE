from utils.BaseFlags import parser as parser

# Dataset
parser.add_argument('--dataset', type=str, default='RobotActions', help="name of the dataset")
parser.add_argument('--num_mods', type=int, default=3, help="number of modalities")
# Experiments
parser.add_argument('--dir_experiment', type=str, default='./experiments', help="directory to save experiment results")

# fid
parser.add_argument('--dir_fid', type=str, default=None, help="directory to save generated samples for fid score calculation")



# Model Architecture
parser.add_argument('--style_dim', type=int, default=32, help="style dimensionality")
parser.add_argument('--num_classes', type=int, default=3, help="number of classes in the dataset")
parser.add_argument('--action_dim', type=int, default=7, help="dimensionality of each action")
parser.add_argument('--dim', type=int, default=128, help="number of units in hidden layers")
parser.add_argument('--num_hidden_layers', type=int, default=2, help="number of hidden layers")
parser.add_argument('--likelihood', type=str, default='normal', help="output distribution")

# Data
parser.add_argument('--data_path', type=str, required=True, help="path to the data .npy file")
parser.add_argument('--train_ratio', type=float, default=0.8, help="ratio of data to use for training")

# Multimodal
parser.add_argument('--subsampled_reconstruction', default=True, help="subsample reconstruction path")

# weighting of loss terms
parser.add_argument('--div_weight', type=float, default=None, help="default weight divergence per modality, if None use 1/(num_mods+1).")
parser.add_argument('--div_weight_uniform_content', type=float, default=None, help="default weight divergence term prior, if None use (1/num_mods+1)")



# Annealing
parser.add_argument('--kl_annealing', type=int, default=0, help="number of kl annealing steps; 0 if no annealing should be done")