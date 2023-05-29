"""
Script for image reconstruction inference with pre-trained models
Author: Patrick Ebel (github/PatrickTUM), based on the scripts of 
        Vivien Sainte Fare Garnot (github/VSainteuf)
License: MIT
"""

import os
import sys
import json
import pprint
import argparse
from parse_args import create_parser

import torch

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from src import utils
from src.model_utils import get_model, load_checkpoint
from train_reconstruct import iterate, save_results, prepare_output, import_from_path, seed_packages
from data.dataLoader import SEN12MSCR, SEN12MSCRTS, get_pairedS1

from torch.utils.tensorboard import SummaryWriter

parser = create_parser(mode='test')
test_config = parser.parse_args()

# grab the PID so we can look it up in the logged config for server-side process management
test_config.pid = os.getpid()

# related to flag --use_custom:
# define custom target S2 patches (these will be mosaiced into a single sample), and fetch associated target S1 patches as well as input data
# (TODO: keeping this hard-coded until a more convenient way to pass it as an argument comes about ...)
targ_s2 = [f'ROIs1868/73/S2/14/s2_ROIs1868_73_ImgNo_14_2018-06-21_patch_{pdx}.tif' for pdx in [171,172,173, 187,188,189, 203,204,205]]

# load previous config from training directories

# if no custom path to config file is passed, try fetching config file at default location
conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json") if not test_config.load_config else test_config.load_config
if os.path.isfile(conf_path):
    with open(conf_path) as file:
        model_config = json.loads(file.read())
        t_args = argparse.Namespace()
        # do not overwrite the following flags by their respective values in the config file
        no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'root1', 'root2', 'root3', 
        'max_samples_count', 'batch_size', 'display_step', 'plot_every', 'export_every', 'input_t', 'region', 'min_cov', 'max_cov']
        conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
        for key, val in vars(test_config).items(): 
            if key in no_overwrite: conf_dict[key] = val
        t_args.__dict__.update(conf_dict)
        config = parser.parse_args(namespace=t_args)
else: config = test_config # otherwise, keep passed flags without any overwriting
config = utils.str2list(config, ["encoder_widths", "decoder_widths", "out_conv"])

if config.pretrain: config.batch_size = 32

experime_dir = os.path.join(config.res_dir, config.experiment_name)
if not os.path.exists(experime_dir): os.makedirs(experime_dir)
with open(os.path.join(experime_dir, "conf.json"), "w") as file:
    file.write(json.dumps(vars(config), indent=4))

# seed everything
seed_packages(config.rdm_seed)
if __name__ == "__main__": pprint.pprint(config)

# instantiate tensorboard logger
writer = SummaryWriter(os.path.join(config.res_dir, config.experiment_name))


if config.use_custom: 
    print('Testing on custom data samples')
    # define a dictionary for the custom sample, with customized ROI and time points
    custom = [{ 'input':  {'S1': [get_pairedS1(targ_s2, config.root1, mod='s1', time=tdx) for tdx in range(0,3)], 
                           'S2': [get_pairedS1(targ_s2, config.root1, mod='s2', time=tdx) for tdx in range(0,3)]},
                'target': {'S1': [get_pairedS1(targ_s2, config.root1, mod='s1')], 'S2': [targ_s2]}}]

def main(config):
    device = torch.device(config.device)
    prepare_output(config)

    model = get_model(config)
    model = model.to(device)
    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)
    
    # get data loader
    if config.pretrain:
        dt_test        = SEN12MSCR(os.path.expanduser(config.root3), split='test', region=config.region, sample_type=config.sample_type)
    else:
        imported_path  = None if any((config.min_cov!=0, config.max_cov!=1)) else import_from_path('test', config)
        dt_test        = SEN12MSCRTS(os.path.expanduser(config.root2), split='test', region=config.region, sample_type=config.sample_type , n_input_samples=config.input_t, 
                                import_data_path=imported_path, sampler='fixed', custom_samples=None if not config.use_custom else custom)
    
    dt_test     = torch.utils.data.Subset(dt_test, range(0, min(config.max_samples_count, len(dt_test))))
    test_loader = torch.utils.data.DataLoader(dt_test, batch_size=config.batch_size, shuffle=False)

    # Load weights
    ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
    load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")

    # Inference
    print("Testing . . .")
    model.eval()

    _, test_img_metrics = iterate(model, data_loader=test_loader, config=config, writer=writer,
                            mode="test", epoch=1, device=device)
    print(f'\nTest image metrics: {test_img_metrics}')

    save_results(test_img_metrics, os.path.join(config.res_dir, config.experiment_name), split='test')
    print(f'\nLogged test metrics to path {os.path.join(config.res_dir, config.experiment_name)}')    


if __name__ == "__main__":
    main(config)
    exit()