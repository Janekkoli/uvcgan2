#!/usr/bin/env python3

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


def multiply_images(synth_path, background_path):
    with Image.open(synth_path) as image:
        synth = image.convert('RGB')
    with Image.open(background_path) as image:
        background = image.convert('RGB').resize(synth.size, Image.BILINEAR)

    synth_pixels = np.asarray(synth, dtype=np.float32)
    background_pixels = np.asarray(background, dtype=np.float32)
    output = np.rint(synth_pixels * background_pixels / 255.0)
    return Image.fromarray(output.clip(0, 255).astype(np.uint8), mode='RGB')


def make_dataset(source, backgrounds_catalog, output_directory, seed=None):
    source = Path(source)
    backgrounds_catalog = Path(backgrounds_catalog)
    output_directory = Path(output_directory)

    for directory in (source, backgrounds_catalog):
        if not directory.is_dir():
            raise FileNotFoundError(f'Directory does not exist: {directory}')

    if output_directory.resolve() in {
        source.resolve(), backgrounds_catalog.resolve()
    }:
        raise ValueError('Output directory must differ from both input directories')

    background_files = image_files(backgrounds_catalog)
    if not background_files:
        raise ValueError(f'No supported background images found in {backgrounds_catalog}')

    for split in ('train', 'val'):
        for domain in ('synth', 'real'):
            input_dir = source / split / domain
            if not input_dir.is_dir():
                raise FileNotFoundError(f'Missing dataset directory: {input_dir}')
        synth_files = image_files(source / split / 'synth')
        if not synth_files:
            raise ValueError(f'No supported synthetic images found in {source / split / "synth"}')
        output_names = [f'{path.stem}.png' for path in synth_files]
        if len(output_names) != len(set(output_names)):
            raise ValueError(
                f'Synthetic images have duplicate filename stems in {source / split / "synth"}'
            )

    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f'Output directory is not empty: {output_directory}'
        )

    rng = random.Random(seed)
    output_directory.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val'):
        for path in image_files(source / split / 'real'):
            real_output = output_directory / split / 'real'
            real_output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, real_output / path.name)

        synth_output = output_directory / split / 'synth'
        synth_output.mkdir(parents=True, exist_ok=True)
        for synth_path in image_files(source / split / 'synth'):
            background_path = rng.choice(background_files)
            multiply_images(synth_path, background_path).save(
                synth_output / f'{synth_path.stem}.png'
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create a dataset with train/val synth and real folders, multiplying '
            'each synthetic image by a randomly selected background.'
        )
    )
    parser.add_argument('dataset', type=Path)
    parser.add_argument('backgrounds_catalog', type=Path)
    parser.add_argument('output_directory', type=Path)
    parser.add_argument(
        '--seed', type=int,
        help='Optional random seed for repeatable background assignments.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    make_dataset(
        args.dataset,
        args.backgrounds_catalog,
        args.output_directory,
        args.seed,
    )