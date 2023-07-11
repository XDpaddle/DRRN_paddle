import paddle
import argparse
import glob
import h5py
import numpy as np
from utils import load_image
from utils import modcrop
from utils import generate_lr
from utils import generate_patch
from utils import image_to_array
from utils import rgb_to_y
from utils import normalize
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=31)
    parser.add_argument('--stride', type=int, default=21)
    args = parser.parse_args()
    hr_patches = []
    lr_patches = []
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.
        images_dir)))):
        hr = load_image(image_path)
        hr = modcrop(hr, 2)
        lr = generate_lr(hr, 2)
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            hr_patches.append(patch)
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            lr_patches.append(patch)
        hr = load_image(image_path)
        hr = modcrop(hr, 3)
        lr = generate_lr(hr, 3)
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            hr_patches.append(patch)
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            lr_patches.append(patch)
        hr = load_image(image_path)
        hr = modcrop(hr, 4)
        lr = generate_lr(hr, 4)
        for patch in generate_patch(hr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            hr_patches.append(patch)
        for patch in generate_patch(lr, args.patch_size, args.stride):
            patch = image_to_array(patch)
            patch = np.expand_dims(normalize(rgb_to_y(patch.astype(np.
                float32), 'chw')), 0)
            lr_patches.append(patch)
        print('Images: {}, Patches: {}'.format(i + 1, len(hr_patches)))
    h5_file = h5py.File(args.output_path, 'w')
    h5_file.create_dataset('hr', data=np.array(hr_patches))
    h5_file.create_dataset('lr', data=np.array(lr_patches))
    h5_file.close()
