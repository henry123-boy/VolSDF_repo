# VolSDF_repo

**This project is a Pytorch implementation of VolSDF**

**Update** the extension of instant ngp was added 

# Data preparation
In this project, we reproduced the result of VolSDF in DTU. And the we got the processed data from https://s3.eu-central-1.amazonaws.com/avg-projects/unisurf/data/DTU.zip

With the structure for files organization below:

    |dataset
        |DTU
        |scan_x
            |scan
            |image
            |mask
            |camera.npz
            |test.lst
            |train.lst
            |val.lst
        
You are expected to create a new director ./dataset under the root folder.
# Build up the Environment
You can build up the required Environment by the command below:

        conda env create -f VolSDF.yaml
        conda activate VolSDF-env
        
# Train the model on the DTU dataset

        python train.py --group=<exp_group_name> --model=VolSDF --yaml=VolSDF_DTU --name=<exp_name> --data.scene=<scan_x>        # scan_x represents the scan you want to train
        
# Testing and other illustrations
+ **We split a scene (49 or 65 pictures in DTU) into a proportion of 0.9 percent training data and 0.1 test data**. The val_ratio can be changed in VolSDF_DTU.yaml
+ We tested the model by the images of the resolution of [150,200], while trained it by the images of the resolution of [1200,1600]
+ **Mesh visualization will be executed according to the frequence of "mesh_vis" written in VolSDF_DTU.yaml, which will be saved with the format of ".ply"**

