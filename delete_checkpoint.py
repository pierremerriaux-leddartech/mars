import os
import glob
folder = '/home/pierre.merriaux/project/mars-refact/mars-debug2/NSG-Studio/outputs/kitti-full-a10-latents-opt/kitti-full-carnerf/2024-01-04_205114/nerfstudio_models/'
# folder = '/home/pierre.merriaux/project/mars-refact/mars/outputs/Mars_pandaset-011-pierre-50frames/mars-pandaset-car-depth-recon/2023-12-08_015800/nerfstudio_models/'

checkpoints_files = glob.glob(os.path.join(folder, '*.ckpt'))
checkpoints_files = sorted(checkpoints_files)
print(checkpoints_files)

for file in checkpoints_files[:-1]:
    idx = int(os.path.basename(file).split('.')[0].split('-')[1])

    if idx % 5000 != 0:
        print(file)
        os.remove(file)


checkpoints_files = glob.glob(os.path.join(folder, '*.ckptlatents'))
checkpoints_files = sorted(checkpoints_files)
print(checkpoints_files)

for file in checkpoints_files[:-1]:
    idx = int(os.path.basename(file).split('.')[0].split('-')[2])

    if idx % 5000 != 0:
        print(file)
        os.remove(file)
