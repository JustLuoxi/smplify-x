# %load_ext autoreload
# %autoreload 2
# %matplotlib notebook
# %matplotlib inline

expr_dir = './vposer_v1_0' #'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
bm_path =  '../smplx/SMPLX_MALE.npz'#'PATH_TO_SMPLX_model.npz'  obtain from https://smpl-x.is.tue.mpg.de/downloads


from human_body_prior.body_model.body_model_vposer import BodyModelWithPoser


bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, poser_type='vposer', smpl_exp_dir=expr_dir).to('cuda')


print('poZ_body', bm.poZ_body.shape)
print('pose_body', bm.pose_body.shape)

from human_body_prior.tools.model_loader import load_vposer

vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.visualization_tools import imagearray2file
from notebooks.notebook_tools import show_image
from notebooks.sample_body_pose import dump_vposer_samples

import numpy as np
bm = BodyModel(bm_path)
num_poses = 5 # number of body poses in each batch

sampled_pose_body = vposer_pt.sample_poses(num_poses=num_poses) # will a generate Nx1x21x3 tensor of body poses
# expr_dir: directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads
images = dump_vposer_samples(bm, sampled_pose_body,out_imgpath='./savefolder/testvposer.png',save_ply=True)
# img = imagearray2file(images)
# show_image(np.array(img)[0])
