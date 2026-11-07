tmux attach -t train


# Ewaluacja
python3 scripts/translate_images.py outdir/crystalSmall/model100epochs --split val

/lu/tetyda/home/jk426853/miniforge3/envs/uvcgan2/bin/python scripts/translate_images.py /lu/tetyda/home/jk426853/uvcgan2/outdir/crystalSmall/crystalSmall_chatFix/model5 --split val
# Downsize
python3 ./scripts/downsize_right.py -s 64 64 -i lanczos /home/jan/uvcgan2/uvcgan2/data/573_files /home/jan/uvcgan2/uvcgan2/data/backgroundSmall
# Pretrain
python3 scripts/crystall/pretrain_crystal --batch_size 45

# Train

python3 scripts/crystal/train_crystal_translation.py --batch-size 32


# Dirs
Backgorunds:
/lu/tetyda/projects/GKW/PLB/background_images_573_files

/lu/tetyda/projects/GKW/PLB/experimental_images/crop_2957

Dataset: 
/lu/tetyda/home/jk426853/MIMUW_Licencjat_Mat/crystal_data/data/crystal_dataset

conda env: 
uvcgan2              *   /lu/topola/home/jk426853/miniforge3/envs/uvcgan2


# Topolo python:
/lu/tetyda/home/jk426853/miniforge3/envs/uvcgan2/bin/python



# Dodawanie .batch do kolejki
sbatch pretrain.batch

squeue -u $USER

tail -f logs/job_214962.err

scancel 214960

srun --jobid=214962 nvidia-smi


# Adding backgrounds
/lu/tetyda/home/jk426853/miniforge3/envs/uvcgan2/bin/python scripts/makeDataWithBackground.py /lu/tetyda/home/jk426853/uvcgan2/data/crystal160 /lu/tetyda/projects/GKW/PLB/background_images_573_files/573_files /lu/tetyda/home/jk426853/uvcgan2/data/crystal160withBackgrounds


# Usefull
for f in *0010_*; do mv "$f" "${f//0010_/}"; done