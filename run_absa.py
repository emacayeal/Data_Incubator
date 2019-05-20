from util.funcs import join, run_absa_argparser, get_absa_save_dir
from util.train import train_absa
from util.inference import pred_absa
from networks.models import base_lstm, td_lstm
import yaml
import sys
import os

try:
    config_file = sys.argv[1]
    if 'yaml' not in config_file:
        config_file = './configs/run_absa_config.yaml'
except IndexError:
    config_file = './configs/run_absa_config.yaml'

yaml.add_constructor('!join', join)
with open(config_file, 'r') as yml_file:
    absa_cfg = yaml.load(yml_file)

# if inferring only, load config and weights from a pretrained directory
if not absa_cfg['run']['run_train'] and absa_cfg['run']['run_infer']:
    absa_cfg = yaml.load(absa_cfg['run']['run_infer_dir']+'run_absa_config.yaml')
    absa_cfg['run']['run_train'] = False
    absa_cfg['run']['run_infer'] = True
    absa_cfg[absa_cfg['network']+'_args']['weights_path'] = absa_cfg['run']['run_infer_dir'] + 'final_weights.h5'

# can load in command line args here in the future

# create output dir and save the config there
out_dir = get_absa_save_dir(absa_cfg)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

net_cfg = absa_cfg['network']
run_cfg = absa_cfg['run']
inp_cfg = absa_cfg['inp']
out_cfg = absa_cfg['out']
out_cfg['out_dir'] = out_dir

network = net_cfg['network']

assert network in net_cfg['network_options'],\
    'Chosen network %s is not in net options %s' % (network, ', '.join(map(str, net_cfg['network_options'])))
assert network in run_cfg,\
    'Chosen network %s is not in run options %s' % (network, ', '.join(map(str, run_cfg['run_options'])))
assert network in inp_cfg,\
    'Chosen network %s is not in input options %s' % (network, ', '.join(map(str, inp_cfg['input_options'])))

if network == 'base_lstm':
    net = base_lstm(**net_cfg[network+'_args'])
elif network == 'td_lstm':
    net = td_lstm(**net_cfg[network+'_args'])

if run_cfg['run_train']:
    print('Getting ready to train')
    net = train_absa(net, run_cfg, network, inp_cfg, out_cfg)

if run_cfg['run_infer']:
    print('Getting ready to run inference')
    pred_absa(net, run_cfg, network, inp_cfg, out_cfg)