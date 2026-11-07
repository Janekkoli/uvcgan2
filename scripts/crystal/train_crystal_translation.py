import argparse
import os

from uvcgan2               import ROOT_OUTDIR, train
from uvcgan2.presets       import GEN_PRESETS, BH_PRESETS
from uvcgan2.utils.parsers import (
    add_preset_name_parser, add_batch_size_parser
)

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Train Crystal real-to-synthetic I2I model'
    )

    add_preset_name_parser(parser, 'gen',  GEN_PRESETS, 'uvcgan2')
    add_preset_name_parser(parser, 'head', BH_PRESETS,  'bn', 'batch head')
    add_batch_size_parser(parser, default = 1)

    parser.add_argument(
        '--no-pretrain', dest = 'no_pretrain', action = 'store_true',
        help = 'disable usage of the pre-trained generator'
    )

    parser.add_argument(
        '--lambda-gp', dest = 'lambda_gp', type = float,
        default = 0.01, help = 'magnitude of the gradient penalty'
    )

    parser.add_argument(
        '--lambda-cycle', dest = 'lambda_cyc', type = float,
        default = 10.0, help = 'magnitude of the cycle-consistency loss'
    )

    parser.add_argument(
        '--lr-gen', dest = 'lr_gen', type = float,
        default = 5e-5, help = 'learning rate of the generator'
    )

    return parser.parse_args()

def get_transfer_preset(cmdargs):
    if cmdargs.no_pretrain:
        return None

    # base_model = (
    #     'crystalSmall/'
    #     'model_m(autoencoder)_d(None)'
    #     f"_g({GEN_PRESETS[cmdargs.gen]['model']})_pretrain-{cmdargs.gen}"
    # )

    base_model = (
    'crystalSmall/'
    'PretrainedwithBackgrounds'
)
    return {
        'base_model' : base_model,
        'transfer_map'  : {
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
                    'path'   : 'crystal160withBackgrounds',
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
        'workers'    : 1,
    },
    'epochs'      : 500,
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 5e-5,  # ZMIANA: Zmniejszono z 1e-4, aby osłabić zbyt silny dyskryminator
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : cmdargs.lr_gen,  # Zalecana wartość w cmdargs to 1e-4 lub 5e-5
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2',
    'model_args' : {
        'lambda_a'        : 10.0,  # ZMIANA: Zwiększenie wagi cyklu A (jeśli cmdargs.lambda_cyc było < 10)
        'lambda_b'        : 10.0,  # ZMIANA: Zwiększenie wagi cyklu B (jeśli cmdargs.lambda_cyc było < 10)
        'lambda_idt'      : 0.5,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    
    'loss'            : 'lsgan',
    'steps_per_epoch' : 500,  # ZMIANA: Zmniejszono z 2000 na 500
    'transfer'        : get_transfer_preset(cmdargs),
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_({cmdargs.no_pretrain}'
        f':{cmdargs.lambda_cyc}:{cmdargs.lambda_gp}:{cmdargs.lr_gen}'
        f':bs{cmdargs.batch_size})'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'crystalSmall', 'crystalSmall'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 10,
}


args_dict_v2 = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : domain,
                    'path'   : 'crystal160withBackgrounds',
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
        'workers'    : 1,
    },
    'epochs'      : 50,  # Ustaw wybraną liczbę epok (np. 30-50)
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 2.5e-5,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.01,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 1e-4,
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.01,
        },
    },
    'model' : 'uvcgan2',
    'model_args' : {
        'lambda_a'        : 15.0,
        'lambda_b'        : 15.0,
        'lambda_idt'      : 1.0,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : 10.0,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    'loss'            : 'lsgan',
    'steps_per_epoch' : 400,
    'transfer'        : get_transfer_preset(cmdargs),
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_v2_ttur_lam15'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'crystalSmall', 'crystalSmall_v2'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,  # Zapisywanie checkpointów co 5 epok dla szybszej weryfikacji
}


args_dict_v3 = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'image-domain-hierarchy',
                    'domain' : domain,
                    'path'   : 'crystal160withBackgrounds',
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
        'workers'    : 1,
    },
    'epochs'      : 50,
    'discriminator' : {
        'model'      : 'basic',
        'model_args' : { 'shrink_output' : False, },
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 5e-5,   # Osłabiony dyskryminator
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
        'spectr_norm' : True,
    },
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'Adam',
            'lr'    : 2e-4,   # Agresywniejszy generator (4x szybszy od dyskryminatora)
            'betas' : (0.5, 0.99),
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        },
    },
    'model' : 'uvcgan2',
    'model_args' : {
        'lambda_a'        : 10.0,
        'lambda_b'        : 10.0,
        'lambda_idt'      : 0.5,
        'avg_momentum'    : 0.9999,
        'head_queue_size' : 3,
        'head_config'     : {
            'name'            : BH_PRESETS[cmdargs.head],
            'input_features'  : 512,
            'output_features' : 1,
            'activ'           : 'leakyrelu',
        },
    },
    'gradient_penalty' : {
        'center'    : 0,
        'lambda_gp' : cmdargs.lambda_gp,
        'mix_type'  : 'real-fake',
        'reduction' : 'mean',
    },
    'loss'            : 'lsgan',
    'steps_per_epoch' : 300,  # Tylko 300 kroków na epokę – bardzo szybki podgląd wyników
    'transfer'        : get_transfer_preset(cmdargs),
    'label'  : (
        f'{cmdargs.gen}-{cmdargs.head}_v3_fast_lr2e4'
    ),
    'outdir' : os.path.join(ROOT_OUTDIR, 'crystalSmall', 'crystalSmall_v3'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 5,
}

train(args_dict_v3)
