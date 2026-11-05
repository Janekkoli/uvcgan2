tmux attach -t train


# Ewaluacja
python3 scripts/translate_images.py outdir/crystalSmall/model100epochs --split val
# Downsize
python3 ./scripts/downsize_right.py -s 64 64 -i lanczos /home/jan/uvcgan2/uvcgan2/data/573_files /home/jan/uvcgan2/uvcgan2/data/backgroundSmall
# Pretrain
python3 scripts/crystall/pretrain_crystall --batch_size 45

# Train

python3 scripts/crystal/train_crystal_translation.py --batch-size 32