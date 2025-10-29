import os
import argparse
import numpy as np
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R 

def convert_smpl_to_smplx(input_path, output_path, gender='neutral'):
    # Load SMPL data
    smpl_data = np.load(input_path, allow_pickle=True)
    data_dict = dict(smpl_data)  # Convert to dict for modification

    # Handle betas padding for SMPL-X (pad from 10 to 16 if necessary)
    if 'betas' in data_dict:
        betas = data_dict['betas']
        # if betas.shape == (10,):
        #     data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
        #     print(f"Padded betas from 10 to 16 for {input_path}")
        # elif betas.shape not in [(16,), (1, 16)]:
        #     raise ValueError(f"Unexpected betas shape: {betas.shape}. Expected (10,), (16,), or (1,16) for padding to SMPL-X.")
        if betas.shape == (10,):
            data_dict['betas'] = np.concatenate([betas, np.zeros(6, dtype=betas.dtype)])
        elif betas.shape == (300,) or betas.shape == (1, 300):
            # Take only the first 10 shape parameters for SMPL-X
            # data_dict['betas'] = np.concatenate([betas[:10], np.zeros(6, dtype=betas.dtype)])
            data_dict['betas'] = betas[:16]
        else:
            raise ValueError(f"Unexpected betas shape: {betas.shape}. Expected (10,), (16,), (300,), or (1,16) for padding to SMPL-X.")

    # Handle mocap_frame_rate variations
    if 'mocap_framerate' in data_dict:
        data_dict['mocap_frame_rate'] = data_dict.pop('mocap_framerate')
        print(f"Renamed 'mocap_framerate' to 'mocap_frame_rate' for {input_path}")

    if 'poses' not in data_dict:
        raise ValueError("Input file does not contain 'poses' key. Is this an SMPL file?")

    poses = data_dict['poses']
    if poses.shape[1] > 72:
        poses = poses[:, :72]

    # Map to SMPL-X format
    data_dict['root_orient'] = poses[:, :3]

    # Rotation adjustment
    # Convert axis-angle to rotation matrix
    rotation = R.from_rotvec(data_dict['root_orient'])
    correction = R.from_euler('x', 90, degrees=True) # SMPL-X (Y-up) to Unity (Z-up): rotate +90° around X
    corrected_rotation = correction * rotation
    data_dict['root_orient'] = corrected_rotation.as_rotvec()

    # 2. Fix translation (Y-up to Z-up)
    trans = data_dict['trans']
    if len(trans.shape) == 1:
        # [X, Y, Z] -> [X, Z, Y] with possible sign changes
        data_dict['trans'] = np.array([trans[0], trans[2], trans[1]])
    else:
        # For multiple frames: [X, Y, Z] -> [X, Z, Y]
        data_dict['trans'] = np.stack([
            trans[:, 0],  # X stays same
            trans[:, 2],  # Z becomes new Y
            trans[:, 1]   # Y becomes new Z
        ], axis=1)

    data_dict['pose_body'] = poses[:, 3:66]  # 21 joints x 3 = 63, ignoring SMPL hand poses

    # Compute ground offset directly here
    # ground_offset = np.min(data_dict['trans'][:, 2])/2.0
    ground_offset = 0.32
    print(f"Ground offset {ground_offset}")
    data_dict['trans'][:, 2] -= ground_offset


    # Ensure gender is set
    if 'gender' not in data_dict:
        data_dict['gender'] = np.array(gender)

    # Remove original poses key
    del data_dict['poses']

    # Save as SMPL-X npz
    np.savez(output_path, **data_dict)
    print(f"Converted {input_path} to {output_path}")

def process_directory(src_folder, tgt_folder, gender='neutral'):
    os.makedirs(tgt_folder, exist_ok=True)
    for filename in tqdm(os.listdir(src_folder)):
        if filename.endswith('.npz'):
            input_path = os.path.join(src_folder, filename)
            output_path = os.path.join(tgt_folder, filename)
            convert_smpl_to_smplx(input_path, output_path, gender)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMPL motion data to SMPL-X format.")
    parser.add_argument("--src_folder", type=str, help="Source directory of SMPL .npz files")
    parser.add_argument("--tgt_folder", type=str, help="Target directory for SMPL-X .npz files")
    parser.add_argument("--input_file", type=str, help="Single input SMPL .npz file")
    parser.add_argument("--output_file", type=str, help="Single output SMPL-X .npz file")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"],
                        help="Gender for SMPL-X model if not present in file.")
    args = parser.parse_args()

    if args.src_folder and args.tgt_folder:
        process_directory(args.src_folder, args.tgt_folder, args.gender)
    elif args.input_file and args.output_file:
        convert_smpl_to_smplx(args.input_file, args.output_file, args.gender)
    else:
        parser.print_help()
