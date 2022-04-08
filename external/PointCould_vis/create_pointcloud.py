'''
this visualization codes are adapted from https://github.com/Brummi/MonoRec
'''

from pathlib import Path
import torch
from .utils.ply_utils import PLYSaver
import pdb
def create_pointcloud(opt,images,depths,poses,intrinsics):
    # create or open output folder
    output_dir = Path("{}/point_cloud".format(opt.output_path))
    output_dir.mkdir(exist_ok=True, parents=True)

    # loading the setting for the visualization
    file_name = "{}_pc.ply".format(opt.data.scene)
    roi = None
    max_d = opt.nerf.depth.max_d
    min_d = opt.nerf.depth.min_d

    # create plysaver
    image_size = images[0].shape[2:4]
    batch_size=images[0].shape[0]
    plysaver = PLYSaver(image_size[0], image_size[1], min_d=min_d, max_d=max_d, batch_size=batch_size, roi=roi, dropout=.75)
    plysaver.to(images[0].device)

    with torch.no_grad():
        for (depth,image,intrinsic,pose) in zip(depths,images,intrinsics,poses):
            plysaver.add_depthmap(depth, image, intrinsic, pose)
        with open(output_dir / file_name, "wb") as f:
            plysaver.save(f)
