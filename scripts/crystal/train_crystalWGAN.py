import argparse
import os

from uvcgan2 import ROOT_OUTDIR, train
from uvcgan2.presets import GEN_PRESETS
from uvcgan2.utils.parsers import add_preset_name_parser, add_batch_size_parser


def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train Crystal translation model with WGAN-GP'
    )
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'uvcgan2')
    add_batch_size_parser(parser, default = 32)

    parser.add_argument(
        '--no-pretrain', dest = 'no_pretrain', action = 'store_true',
        help = 'disable usage of the pre-trained generator'
    )
    parser.add_argument(
        '--lambda-gp', dest = 'lambda_gp', type = float,
        default = 10.0, help = 'magnitude of the WGAN-GP penalty'
    )
    parser.add_argument(
        '--lambda-cycle', dest = 'lambda_cyc', type = float,
        default = 10.0, help = 'magnitude of the cycle-consistency loss'
    )
    parser.add_argument(
        '--lr-gen', dest = 'lr_gen', type = float,
        default = 1e-4, help = 'generator learning rate'
    )
    parser.add_argument(
        '--lr-disc', dest = 'lr_disc', type = float,
        default = 1e-4, help = 'critic learning rate'
    )

    return parser.parse_args()


def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    return {
        'base_model' : 'crystalSmall/BertFirstonlysynththen60normal',
        'transfer_map' : {
            'gen_ab' : 'encoder',
            'gen_ba' : 'encoder',
        },
        'strict'        : True,
        'allow_partial' : False,
        'fuzzy'         : None,
    }


cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : domain,
                    'path'   : 'crystal160',
                },
                'shape'           : (3, 160, 160),
                'transform_train' : [
                    { 'name' : 'random-rotation', 'degrees' : 10, },
                    'random-flip-horizontal',
                ],
                'transform_test' : None,
            } for domain in [ 'real', 'synth' ]
        ],
        'merge_type' : 'unpaired',
        'workers'    : 4,
    },
    'epochs'      : 600,
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 3e-5,
            'betas' : (0.0, 0.9),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.0, 0.9),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2',
    'model_args' : {
        'lambda_a'        : cmdargs.lambda_cyc,
        'lambda_b'        : cmdargs.lambda_cyc,
        'lambda_idt'      : 0.15,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'discriminator_steps' : 5, # tutaj sie ustawia stosunek ilosci krokow dyskryminatora do generatora
        'head_config'     : {
            'name'            : 'simple-average',
            'input_features'  : 512,
            'average_spacial' : True,
        },
    },
    'gradient_penalty' : {
        'center'    : 1,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },

    'loss'            : 'wgan',
    'steps_per_epoch' : 500,
    'transfer'        : get_transfer_preset(cmdargs),
    'label'  : (
        f'WGAN-GP-{cmdargs.gen}_({cmdargs.no_pretrain}'
        f':{cmdargs.lambda_cyc}:{cmdargs.lambda_gp}'
        f':{cmdargs.lr_gen}:{cmdargs.lr_disc}'
        f':bs{cmdargs.batch_size})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'crystalSmall', 'crystalWGAN45h'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 15,
}

train(args_dict)
