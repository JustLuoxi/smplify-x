# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os.path as osp
import argparse

import numpy as np
import torch

import smplx
from utils import *

def main(model_folder, model_type='smplx', ext='npz',
         gender='neutral', plot_joints=False,
         plotting_module='pyrender',
         use_face_contour=False):


    # rest pose
    use_hands = False
    use_face = False
    use_face_contour = False
    openpose_format = 'coco25'
    map = smpl_to_openpose(model_type, use_hands=use_hands,
                            use_face=use_face,
                            use_face_contour=use_face_contour,
                            openpose_format=openpose_format)
    joint_mapper = JointMapper(map)

    model = smplx.create(model_folder, model_type=model_type,
                         gender=gender, use_face_contour=use_face_contour,
                         joint_mapper=joint_mapper,
                         ext=ext)
    print(model)

    betas = torch.zeros([1, 10], dtype=torch.float32)
    expression = torch.zeros([1, 10], dtype=torch.float32)
    output = model(betas=betas, expression=expression, joint_mapper = joint_mapper,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().numpy()

    # globalRT
    # load target 3d skeleton
    openpose_folder = 'L:\\data\\exp\\labeling\\mit_bouncing\\45\\openpose\\'
    target_skeleton, bodysize = LoadBodySkeleton(openpose_folder)
    if len(target_skeleton.shape) == 3:
        if target_skeleton.shape[0] > args.batch_size:
            target_skeleton = target_skeleton[:1, :, :]
        else:
            batch_size = target_skeleton.shape[0]
    else:
        target_skeleton = target_skeleton.repeat((1, 1, 1))

    global_rot = ComputeGlobalR(target_skeleton, joints, isBody=True)
    global_rot = torch.tensor(global_rot, dtype=torch.float32, requires_grad=True)
    global_transl = target_skeleton[:,8,:] - torch.from_numpy(joints[:,8,:])
    global_transl = torch.tensor(global_transl, dtype=torch.float32, requires_grad=False)

    output = model(global_orient = global_rot, transl = global_transl,
        betas=betas, expression=expression, joint_mapper=joint_mapper,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    # pose


    ##    export
    import trimesh
    np.savetxt('initial_joints.txt',joints)
    out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    # rot = trimesh.transformations.rotation_matrix(
    #     np.radians(180), [1, 0, 0])
    # out_mesh.apply_transform(rot)
    out_mesh.export('test.obj')

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    # render
    # if plotting_module == 'pyrender':
    #     import pyrender
    #     import trimesh
    #     vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    #     tri_mesh = trimesh.Trimesh(vertices, model.faces,
    #                                vertex_colors=vertex_colors)
    #
    #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)
    #
    #     scene = pyrender.Scene()
    #     scene.add(mesh)
    #
    #     if plot_joints:
    #         sm = trimesh.creation.uv_sphere(radius=0.005)
    #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    #         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
    #         tfs[:, :3, 3] = joints
    #         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    #         scene.add(joints_pcl)
    #
    #     pyrender.Viewer(scene, use_raymond_lighting=True)
    # elif plotting_module == 'matplotlib':
    #     from matplotlib import pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D
    #     from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    #
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     mesh = Poly3DCollection(vertices[model.faces], alpha=0.1)
    #     face_color = (1.0, 1.0, 0.9)
    #     edge_color = (0, 0, 0)
    #     mesh.set_edgecolor(edge_color)
    #     mesh.set_facecolor(face_color)
    #     ax.add_collection3d(mesh)
    #     ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    #
    #     if plot_joints:
    #         ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], alpha=0.1)
    #     plt.show()
    # elif plotting_module == 'open3d':
    #     import open3d as o3d
    #
    #     mesh = o3d.TriangleMesh()
    #     mesh.vertices = o3d.Vector3dVector(
    #         vertices)
    #     mesh.triangles = o3d.Vector3iVector(model.faces)
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color([0.3, 0.3, 0.3])
    #
    #     o3d.visualization.draw_geometries([mesh])
    # else:
    #     raise ValueError('Unknown plotting_module: {}'.format(plotting_module))

def Regressor(**args):


    body_dofs = 63
    shape_ncomps = 10
    pose_ncomps = args.mano_ncomps*2 + body_dofs

    # initialize parameters
    pose_params_cpu = np.zeros([args.batch_size, pose_ncomps])
    shape_params_cpu = np.zeros([args.batch_size, shape_ncomps])
    shape_params_cpu[:] = 0.8
    # global_rot = np.zeros([args.batch_size, 3])

    if args.gender == 'male':
        mano_ske_path = args.mano_root / 'SMPLH_male_SKE.txt'
    elif args.gender == 'female':
        mano_ske_path = args.mano_root / 'SMPLH_female_SKE.txt'
    init_skeleton = torch.from_numpy(np.loadtxt(mano_ske_path)).type(torch.float32)
    init_skeleton = init_skeleton.repeat((args.batch_size, 1, 1))
    global_scale = ComputeGlobalS(target_skeleton, init_skeleton, isBody=True)
    global_rot = ComputeGlobalR(target_skeleton, init_skeleton, isBody=True)
    # bone_id = 1
    # bone_id = bone_id - 1
    # pose_params_cpu[:,bone_id*3:bone_id*3+3] = [1.71, 0, 0]
    pose_params_cpu = ComputeInitPose(target_skeleton, init_skeleton)

    if args.cuda:
        pose_params = torch.tensor(pose_params_cpu, dtype=torch.float32, device='cuda', requires_grad=True)
        shape_params = torch.tensor(shape_params_cpu, dtype=torch.float32, device='cuda', requires_grad=True)
        global_scale = global_scale.cuda()
        global_rot = torch.tensor(global_rot, dtype=torch.float32, device='cuda', requires_grad=True)
        target_skeleton = target_skeleton.cuda()
    else:
        pose_params = torch.tensor(pose_params_cpu, dtype=torch.float32, requires_grad=True)
        shape_params = torch.tensor(shape_params_cpu, dtype=torch.float32, requires_grad=True)
        global_rot = torch.tensor(global_rot, dtype=torch.float32, requires_grad=True)

    # second order optimization
    optimizer_shape = torch.optim.LBFGS([shape_params])
    optimizer_pose = torch.optim.LBFGS([pose_params])
    # optimizer_globalr = torch.optim.LBFGS([global_rot])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='neutral',
                        help='The gender of the model')
    parser.add_argument('--plotting-module', type=str, default='pyrender',
                        dest='plotting_module',
                        choices=['pyrender', 'matplotlib', 'open3d'],
                        help='The module to use for plotting the result')
    parser.add_argument('--ext', type=str, default='npz',
                        help='Which extension to use for loading')
    parser.add_argument('--plot-joints', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='The path to the model folder')
    parser.add_argument('--use-face-contour', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Compute the contour of the face')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module

    main(model_folder, model_type, ext=ext,
         gender=gender, plot_joints=plot_joints,
         plotting_module=plotting_module,
         use_face_contour=use_face_contour)
