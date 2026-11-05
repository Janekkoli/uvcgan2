import argparse
import os

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.presets import GEN_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Pretrain Crystal generators'
    )
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'uvcgan2')
    add_batch_size_parser(parser, default = 32)
    return parser.parse_args()

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : domain,
                    'path'   : 'CrystalSmallBack',
                },
                'shape'           : (3, 64, 64),
                'transform_train' : [
                    { 'name' : 'random-rotation', 'degrees' : 10,  },
                    'random-flip-horizontal',
                    {
                        'name' : 'color-jitter',
                        'brightness' : 0.2,
                        'contrast'   : 0.2,
                        'saturation' : 0.2,
                        'hue'        : 0.2,
                    },
                ],
                'transform_test' : None,
            } for domain in [ 'real', 'synth' ]
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'        : 500,
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'autoencoder',
    'model_args' : {
        'joint'   : True,
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (8, 8),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'    : 'CosineAnnealingWarmRestarts',
        'T_0'     : 500,
        'T_mult'  : 1,
        'eta_min' : cmdargs.batch_size * 5e-8 / 512,
    },
    'transfer': {
    'base_model': 'crystalSmall/Model100epokBertnobackgrounds',
    'transfer_map': {
        'encoder': 'encoder',
    },
    'strict': True,
    'allow_partial': False,
    'fuzzy': None,
    },
    'loss'             : 'l1',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : 'PretrainedNowWithBackgrounds',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'crystalSmall'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 15,
}

train(args_dict)
