import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def image_files(directory):
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def combine_images(synth_path, background_path, mask_path, mode):
    with Image.open(synth_path) as image:
        synth = image.convert('RGB')

    with Image.open(background_path) as image:
        background = image.convert('RGB').resize(synth.size, Image.BILINEAR)

    synth_pixels = np.asarray(synth, dtype=np.float32)
    background_pixels = np.asarray(background, dtype=np.float32)

    if mode == 'multiply':
        output = synth_pixels * background_pixels / 255.0
    else:
        with Image.open(mask_path) as image:
            mask = image.convert('L').resize(synth.size, Image.BILINEAR)
        alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
        output = synth_pixels * alpha + background_pixels * (1.0 - alpha)

    return Image.fromarray(np.rint(output).clip(0, 255).astype(np.uint8), mode='RGB')


def make_dataset(source, backgrounds, output, masks, mode, seed):
    for split in ('train', 'val'):
        for domain in ('real', 'synth'):
            source_dir = source / split / domain
            if not source_dir.is_dir():
                raise FileNotFoundError(f'Missing source directory: {source_dir}')

    background_files = image_files(backgrounds)
    if not background_files:
        raise ValueError(f'No supported background images found in {backgrounds}')

    if mode == 'masked-composite' and masks is None:
        raise ValueError('--masks is required when --mode is masked-composite')

    for split in ('train', 'val'):
        synth_files = image_files(source / split / 'synth')
        if not synth_files:
            raise ValueError(f'No synthetic images found in {source / split / "synth"}')
        if mode == 'masked-composite':
            for synth_path in synth_files:
                mask_path = masks / split / 'synth' / f'{synth_path.stem}.png'
                if not mask_path.is_file():
                    raise FileNotFoundError(f'Missing grayscale mask: {mask_path}')

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f'Output directory is not empty: {output}. '
            'Choose another output path to avoid overwriting data.'
        )

    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'val'):
        real_source = source / split / 'real'
        real_output = output / split / 'real'
        real_output.mkdir(parents=True, exist_ok=True)
        for path in image_files(real_source):
            shutil.copy2(path, real_output / path.name)

        synth_output = output / split / 'synth'
        synth_output.mkdir(parents=True, exist_ok=True)
        for synth_path in image_files(source / split / 'synth'):
            background_path = rng.choice(background_files)
            mask_path = None
            if mode == 'masked-composite':
                mask_path = masks / split / 'synth' / f'{synth_path.stem}.png'

            combined = combine_images(
                synth_path, background_path, mask_path, mode
            )
            combined.save(synth_output / f'{synth_path.stem}.png')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a crystal dataset with synthetic images combined with backgrounds.'
    )
    parser.add_argument('--source', type=Path, default=Path('data/crystalSmall'))
    parser.add_argument('--backgrounds', type=Path, default=Path('data/573_files'))
    parser.add_argument(
        '--output', type=Path, default=Path('data/crystalSmall_synth_bg')
    )
    parser.add_argument(
        '--mode',
        choices=('multiply', 'masked-composite'),
        default='multiply',
        help=(
            'multiply computes synth * background / 255; masked-composite '
            'places the synth image over the background using a grayscale mask.'
        ),
    )
    parser.add_argument(
        '--masks',
        type=Path,
        help='Mask root with <split>/synth/<synthetic-image-stem>.png files.',
    )
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    make_dataset(
        args.source,
        args.backgrounds,
        args.output,
        args.masks,
        args.mode,
        args.seed,
    )
