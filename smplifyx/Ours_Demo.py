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
from tqdm import tqdm

import numpy as np
import torch

import smplx

from smplifyx.utils import *
from smplifyx.loss import *

def main(model_folder, model_type='smplx', ext='npz',
         model = None,
         gender='neutral', plot_joints=False,
         plotting_module='pyrender',
         use_face_contour=False,
         openpose_folder = None,
         output_path = "pose_fitting.ply"):

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
    if model ==None:
        model = smplx.create(model_folder, model_type=model_type,
                             gender=gender, use_face_contour=use_face_contour,
                             joint_mapper=joint_mapper,
                             flat_hand_mean = False,
                             ext=ext)
    print(model)

    betas = torch.zeros([1, 10], dtype=torch.float32, requires_grad=True)
    expression = torch.zeros([1, 10], dtype=torch.float32)
    output = model(betas=betas, expression=expression, joint_mapper = joint_mapper,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().numpy()

    # globalRT
    # load target 3d skeleton
    # openpose_folder = 'L:\\data\\exp\\labeling\\mit_bouncing\\45\\openpose\\'
    # openpose_folder = 'L:\\data\\exp\\labeling\\mit_crane\\0\\openpose\\'
    target_skeleton, bodysize = LoadBodySkeleton(openpose_folder)
    target_skeleton[joints==0]=0
    if len(target_skeleton.shape) == 3:
        if target_skeleton.shape[0] > args.batch_size:
            target_skeleton = target_skeleton[:1, :, :]
        else:
            batch_size = target_skeleton.shape[0]
    else:
        target_skeleton = target_skeleton.repeat((1, 1, 1))

    global_rot = ComputeGlobalR(target_skeleton, joints, isBody=True)
    global_rot = torch.tensor(global_rot, dtype=torch.float32, requires_grad=True)
    global_transl = target_skeleton[:,8,:] -  torch.from_numpy(joints[:,8,:])
    global_transl = torch.tensor(global_transl, dtype=torch.float32, requires_grad=True)

    optimizer_shape = torch.optim.LBFGS([betas])

    # # shape
    # pbar = tqdm(range(1))
    # for idx in pbar:
    #     def closure():
    #         optimizer_shape.zero_grad()
    #         output = model(global_orient = global_rot, transl = global_transl,
    #         betas=betas, expression=expression, joint_mapper=joint_mapper,
    #                    return_verts=True)
    #         loss_shape = BoneLengthLoss(output.joints, target_skeleton, isBody=True)
    #         loss_l2 = JointPositionLoss(output.joints, target_skeleton, bodysize,skipjoints=[0,2,3,4,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    #         loss_shape_reg = ShapeRegLoss(betas)
    #         loss = loss_shape + 20*loss_l2 +0.05 * loss_shape_reg
    #         pbar.set_description("Shape Loss: %.6f, lr: %.6f" % (torch.sum(loss), get_lr(optimizer_shape)))
    #         loss.backward(retain_graph=True)
    #         return loss
    #     optimizer_shape.step(closure)
    #
    # ##    export
    # output = model(global_orient=global_rot, transl=global_transl,
    #                betas=betas,
    #                expression=expression, joint_mapper=joint_mapper, return_verts=True)
    # vertices = output.vertices.detach().cpu().numpy().squeeze()
    # joints = output.joints.detach().cpu().numpy().squeeze()
    # import trimesh
    # np.savetxt('shape_fitting.txt',joints)
    # out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    # out_mesh.export('shape_fitting.obj')

    # pose
    # body_pose_cpu = ComputeInitPose(target_skeleton, output.joints.detach())
    # body_pose_cpu = torch.tensor(body_pose_cpu, dtype=torch.float32, requires_grad=True)

    # load pc:
    # pc_path = 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\40\\mesh_000040.txt'
    # target_pt, target_normal = LoadTargetPC(pc_path)

    body_pose_cpu =  torch.zeros([1, 63], dtype=torch.float32, requires_grad=True)
    target_angle, target_bone_dirs = AnalyzeSkeleton(target_skeleton, isBody=True)
    optimizer_body_pose = torch.optim.LBFGS([body_pose_cpu])
    loss_r = 100
    pbar = tqdm(range(1))
    for idx in pbar:
        def closure():
            optimizer_body_pose.zero_grad()
            optimizer_shape.zero_grad()
            output = model(global_orient = global_rot, transl = global_transl,
                        betas=betas, body_pose = body_pose_cpu,
                           expression=expression, joint_mapper=joint_mapper, return_verts=True)
            output.joints[target_skeleton==0]=0
            loss_shape = BoneLengthLoss(output.joints, target_skeleton, isBody=True)
            loss_l2 = JointPositionLoss(output.joints, target_skeleton, bodysize)
            loss_angle = JointAngleLoss(output.joints, target_angle, target_bone_dirs,isBody=True)
            loss_shape_reg = ShapeRegLoss(betas)
            # loss_dis = PtChamferLoss(output.vertices, target_pt)
            # loss_reg = PoseRegLoss(body_pose_cpu)
            loss = 4 * loss_l2 + 0.1 * loss_angle + loss_shape + 0.05 * loss_shape_reg  # + loss_dis*10
            pbar.set_description("Pose Loss: %.6f, lr: %.6f" % (torch.sum(loss), get_lr(optimizer_body_pose)))
            loss.backward(retain_graph=True)
            loss_r = loss.clone().detach()
            return loss
        optimizer_body_pose.step(closure)
        optimizer_shape.step(closure)

    # output = model(global_orient=global_rot, transl=global_transl,
    #                betas=betas, body_pose=body_pose_cpu,
    #                expression=expression, joint_mapper=joint_mapper, return_verts=True)
    # vertices = output.vertices.clone().detach().cpu().numpy().squeeze()
    # joints = output.joints.clone().detach().cpu().numpy().squeeze()
    # import trimesh
    # np.savetxt(output_path+'_joints.txt',joints)
    # out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    # out_mesh.export(output_path)

    optimizer_t = torch.optim.SGD([global_transl], lr = 0.005, momentum=0.9)
    pbar = tqdm(range(1))
    for idx in pbar:
        def closure():
            optimizer_body_pose.zero_grad()
            optimizer_shape.zero_grad()
            optimizer_t.zero_grad()
            output = model(global_orient = global_rot, transl = global_transl,
                        betas=betas, body_pose = body_pose_cpu,
                           expression=expression, joint_mapper=joint_mapper, return_verts=True)
            output.joints[target_skeleton==0]=0
            loss_shape = BoneLengthLoss(output.joints, target_skeleton, isBody=True)
            loss_l2 = JointPositionLoss(output.joints, target_skeleton, bodysize)
            loss_angle = JointAngleLoss(output.joints, target_angle, target_bone_dirs,isBody=True)
            loss_shape_reg = ShapeRegLoss(betas)
            # loss_dis = PtChamferLoss(output.vertices, target_pt)
            # loss_reg = PoseRegLoss(body_pose_cpu)
            loss = 4*loss_l2 + 0.1*loss_angle + loss_shape + 0.05 * loss_shape_reg #+ loss_dis*10
            pbar.set_description("Pose Loss: %.6f, lr: %.6f" % (torch.sum(loss), get_lr(optimizer_body_pose)))
            loss.backward(retain_graph=True)
            loss_r = loss.clone().detach()
            return loss
        optimizer_body_pose.step(closure)
        optimizer_shape.step(closure)
        optimizer_t.step(closure)

    ##    export
    output = model(global_orient=global_rot, transl=global_transl,
                   betas=betas, body_pose=body_pose_cpu,
                   expression=expression, joint_mapper=joint_mapper, return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()
    import trimesh
    np.savetxt(output_path+'_joints.txt',joints)
    out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
    out_mesh.export(output_path)
    return loss_r

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL-X Demo')

    parser.add_argument('--model-folder', required=True, type=str,
                        help='The path to the model folder')
    parser.add_argument('--model-type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of model to load')
    parser.add_argument('--gender', type=str, default='male',
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
    parser.add_argument('--openpose_folder',type=str,required=True,
                        help='openpose_folder path')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output model path ')

    args = parser.parse_args()

    model_folder = osp.expanduser(osp.expandvars(args.model_folder))
    model_type = args.model_type
    plot_joints = args.plot_joints
    use_face_contour = args.use_face_contour
    gender = args.gender
    ext = args.ext
    plotting_module = args.plotting_module
    openpose_folder = args.openpose_folder
    output_path = args.output_path

    # main(model_folder, model_type, ext=ext,
    #      gender=gender, plot_joints=plot_joints,
    #      plotting_module=plotting_module,
    #      use_face_contour=use_face_contour,
    #      openpose_folder = openpose_folder,
    #      output_path = output_path)

#     run sequence
    # load smplx
    use_hands = True
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

    root_folder = [
        # 'X:\\POSE_EG\\0111_ExperimentData\\D_bouncing\\OURS',
        # 'X:\\POSE_EG\\0111_ExperimentData\\D_handstan\\OURS',
        ## 'X:\\POSE_EG\\0111_ExperimentData\\D_marchaaa\\OURS',
        ## 'X:\\POSE_EG\\0111_ExperimentData\\D_squataaa\\OURS',
        # 'X:\\POSE_EG\\0111_ExperimentData\\I_crane\\OURS',
        # 'X:\\POSE_EG\\0111_ExperimentData\\I_jumpi\\OURS',
        ## 'X:\\POSE_EG\\0111_ExperimentData\\I_march\\OURS',
        ## 'X:\\POSE_EG\\0111_ExperimentData\\I_squat\\OURS',
        # 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS',
        # 'X:\\Just\\OneDrive\\multiviewexp',
        'D:\\POSE\\ICCV\\0_ICCV_EXP\\1-Badminton\\'
    ]

    error_folder = []
    import os

    for root in root_folder:
        # save_folder = root + "\\smplx_fittedmodel_0331"
        # if not os.path.exists(save_folder):
        #     os.mkdir(save_folder)

        for i in range(62, 63, 2):
            save_folder = root + '\\' + str(i) + "\\smplx_fittedmodel_0331"
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

            openpose_folder = root + '\\' + str(i) + "\\openpose"
            skeleton_path = openpose_folder + "\\skeleton_body\\skeleton.txt"
            if not os.path.exists(skeleton_path):
                error_folder.append(skeleton_path)
                continue
            output_path = save_folder + '\\' + str(i).zfill(6) + ".ply"

            if os.path.exists(output_path):
                continue

            loss_r = main(model_folder, model_type, ext=ext, model = model,
                 gender=gender, plot_joints=plot_joints,
                 plotting_module=plotting_module,
                 use_face_contour=use_face_contour,
                 openpose_folder = openpose_folder,
                 output_path = output_path)

            if loss_r>20:
                error_folder.append("large loss: " + openpose_folder)

    with open("error_folder.txt", "w") as f:
        import time
        f.write("------ %s\n ----------" % time.time())
        for item in error_folder:
            f.write("%s \n" % item )

