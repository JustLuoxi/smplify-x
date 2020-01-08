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
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import numpy as np

import torch
import torch.nn as nn

from pathlib import Path
from transform.rotation import Rotation as R

def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


# ---- Just for 3d joints fitting ---
def ComputeGlobalR(target_skeleton_batch, mano_skeleton_batch, isBody=False):
    batch_size = target_skeleton_batch.shape[0]
    global_rot = np.zeros([batch_size, 3])

    for i in range(batch_size):
        mano_skeleton = mano_skeleton_batch[i,:,:]
        target_skeleton = target_skeleton_batch[i,:,:]
        # palm and index
        if isBody:
            smpl_spine = normalizev(mano_skeleton[1] - mano_skeleton[8])
            smpl_shoulder = normalizev(mano_skeleton[2] - mano_skeleton[5])
            smpl_facing = normalizev(np.cross(smpl_spine, smpl_shoulder))

            target_spine = normalizev(target_skeleton[1] - target_skeleton[8])
            target_shoulder = normalizev(target_skeleton[2] - target_skeleton[5])
            target_facing = normalizev(np.cross(target_spine, target_shoulder))

            # compute spine rot from mano to target
            axis = normalizev(np.cross(smpl_spine, target_spine))
            angle = np.arccos(np.dot(smpl_spine, target_spine))
            rot1 = R.from_rotvec(axis * angle)

            smpl_facing_rot1 = R.apply(rot1, smpl_facing)
            axis2 = normalizev(np.cross(smpl_facing_rot1, target_facing))
            angle2 = np.arccos(np.dot(smpl_facing_rot1, target_facing))
            rot2 = R.from_rotvec(axis2 * angle2)
            est_rot = rot2 * rot1

            # fix 180
            rot_mat = est_rot.as_dcm()
            trans_smpl_spine = np.matmul(rot_mat, smpl_spine)
            trans_smpl_shoulder = np.matmul(rot_mat, smpl_shoulder)
            trans_smpl_facing = np.matmul(rot_mat, smpl_facing)

            angle_facing = np.dot(trans_smpl_facing, target_facing)
            angle_spine = np.dot(trans_smpl_spine, target_spine)
            angle_shoulder = np.dot(trans_smpl_shoulder, target_shoulder)

            # rotate index again
            if angle_facing < 0:
                rot3 = R.from_rotvec(axis2*3.1415)
                est_rot = rot3*est_rot
                rot_mat = est_rot.as_dcm()
                trans_smpl_facing = np.matmul(rot_mat, smpl_facing)
                angle_facing = np.dot(trans_smpl_facing, target_facing)

            global_rot[i, :] = est_rot.as_rotvec()

        else:
            # right hand axis
            mano_index = mano_skeleton[9] - mano_skeleton[0]
            mano_pinky = mano_skeleton[17] - mano_skeleton[0]
            mano_palm = np.cross(mano_index, mano_pinky)
            mano_index = normalizev(mano_index)
            mano_palm = normalizev(mano_palm)
            mano_cross = normalizev(np.cross(mano_palm, mano_index))
            mano_indexfix = normalizev(np.cross(mano_palm, mano_cross))

            targ_index = target_skeleton[9] - target_skeleton[0]
            targ_pinky = target_skeleton[17] - target_skeleton[0]
            targ_palm = np.cross(targ_index, targ_pinky)
            targ_index = normalizev(targ_index)
            targ_palm = normalizev(targ_palm)
            targ_cross = normalizev(np.cross(targ_palm, targ_index))

            # compute palm rot from mano to target
            axis = normalizev(np.cross(mano_palm, targ_palm))
            angle = np.arccos(np.dot(mano_palm, targ_palm))
            rot1 = R.from_rotvec(axis * angle)

            mano_indexfix = R.apply(rot1, mano_indexfix)
            axis2 = normalizev(np.cross(mano_indexfix, targ_index))
            angle2 = np.arccos(np.dot(mano_indexfix, targ_index))
            angle2 = angle2
            rot2 = R.from_rotvec(axis2*angle2)
            est_rot = rot2*rot1
            global_rot[i, :] = est_rot.as_rotvec()

            #
            rot_mat = est_rot.as_dcm()
            trans_mano_index = np.matmul(rot_mat, mano_index)
            trans_mano_palm = np.matmul(rot_mat, mano_palm)
            trans_mano_cross = np.matmul(rot_mat, mano_cross)

            angle_index = np.dot(trans_mano_index, targ_index)
            angle_palm = np.dot(trans_mano_palm, targ_palm)
            angle_cross = np.dot(trans_mano_cross, targ_cross)

            # rotate index again
            if angle_index < 0:
                rot3 = R.from_rotvec(axis2*3.1415)
                est_rot = rot3*est_rot
                rot_mat = est_rot.as_dcm()
                trans_mano_index = np.matmul(rot_mat, mano_index)
                angle_index = np.dot(trans_mano_index, targ_index)

            global_rot[i, :] = est_rot.as_rotvec()

    return global_rot

def normalizev(vec):
    if isinstance(vec, np.ndarray):
        return vec/np.linalg.norm(vec)
    else:
        return vec/torch.norm(vec)

def LoadTargetPC(target_path):
    if Path(target_path).exists():
        pt_data = np.loadtxt(target_path)
        pt_normal = np.zeros([1, 3])
        if pt_data.shape[1] == 6:
            pt_cloud = pt_data[:, 0:3]
            pt_normal = pt_data[:,3:]
        pt_cloud = torch.from_numpy(pt_cloud).type(torch.float32)
        pt_normal = torch.from_numpy(pt_normal).type(torch.float32)

    else:
        pt_cloud = torch.zeros(1,3)
        pt_normal = torch.zeros(1, 3)
    return pt_cloud, pt_normal

def LoadBodySkeleton(openpose_folder):
    body_joints_path = openpose_folder + '\\skeleton_body\\skeleton.txt'
    skeleton_body = np.loadtxt(body_joints_path)
    skeleton_all = skeleton_body.copy()
    skeleton_all = torch.from_numpy(skeleton_all).type(torch.float32)
    bodysize = torch.max(torch.cdist(skeleton_all, skeleton_all)) * 0.3
    return skeleton_all, bodysize

def AnalyzeSkeleton(target_verts, isBody=True):
    # target_verts: batch_size, 21, 3
    batch_size = target_verts.shape[0]
    links = torch.tensor([
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    ])
    bone_num = 15
    if isBody:
        links = torch.tensor([
            (0, 1, 8),
            (1,8,10),
            (1,8,13),
            (8,10,11),
            (8,13,14),
            (2, 3, 4),
            (5, 6, 7),
            (10, 11,22),
            (13, 14,19),
            (2, 1, 5),
            (1,0,16),
            (0, 16,18),
            (1,0,15),
            (0,15,17),
        ])
        bone_num = 14
    tar_angles = torch.zeros((batch_size, bone_num))
    tar_bone_directions = torch.zeros((batch_size, bone_num,3)) # discard last bone
    itr = 0
    zero = torch.zeros_like(target_verts[:, 0, :])
    for link in links:
        for j1, j2, j3 in zip(link[0:-2], link[1:-1], link[2:]):
            if torch.equal(target_verts[:, j1, :], zero) or torch.equal(target_verts[:, j2, :],zero):
                tar_angles[:, itr] = 0
                tar_bone_directions[:, itr, :] = 0
                continue
            tar_vec = target_verts[:, j1, :] - target_verts[:, j2, :]  # batch_size * 3
            tar_vec_norm = tar_vec.pow(2).sum(dim=1).pow(0.5)  # batch_size * 1
            tar_vec2 = target_verts[:, j3, :] - target_verts[:, j2, :]
            tar_vec2_norm = tar_vec2.pow(2).sum(dim=1).pow(0.5)
            tar_angle = (tar_vec * tar_vec2).sum(dim=1) / (tar_vec_norm * tar_vec2_norm)  # batch_size
            tar_angles[:, itr] = torch.acos(tar_angle)
            tar_bone_directions[:, itr, :] = -tar_vec
            itr = itr + 1

    return tar_angles, tar_bone_directions

def ComputeInitPose(target_skeleton, init_skeleton, isBody = True):
    target_skeleton = target_skeleton.squeeze(0)
    init_skeleton = init_skeleton.squeeze(0)
    # Input: target_skeleton is openpose style, body + left hand + right hand [65*3]
    # Output: pose_param: axis angle for all bones [51*3]
    # 21 for body
    # 15 for each hand

    # I need to know SMPL kinematic chain
    bone_list = [
        # from , to
        [12,13],    #1 left thigh
        [9,10],    #2 right thigh
        [8,1],     #3 belly
        [13,14],   #4 left lower leg
        [10,11],   #5 right lower leg
        [8,1],     #6 chest
        [14,19],   #7 left foot
        [11,22],   #8 right foot
        [8,1],     #9 upper chest
        [14,19],   #10 left toe
        [11,22],   #11 right toe
        [1,0],     #12 neck
        [1,5],     #13 left shoulder
        [1,2],     #14 right shoulder
        [1,0],     #15 head
        [5,6],     #16 left upper arm
        [2,3],     #17 right upper arm
        [6,7],     #18 left lower arm
        [3,4],     #19 left upper arm
        # [7,33],    #20 wrist - hand left index
        # [4,53]     #21 wrist - hand right index
    ]
    body_pose_param = torch.zeros(21,3)
    counter = 0

    for bone in bone_list:
        smpl_dir = normalizev(init_skeleton[bone[1]] - init_skeleton[bone[0]])
        # smpl_dir = R.apply(rot, mano_indexfix)
        target_dir = normalizev(target_skeleton[bone[1]] - target_skeleton[bone[0]])
        axis = normalizev(np.cross(smpl_dir, target_dir))
        angle = np.arccos(np.dot(smpl_dir, target_dir))
        rot = R.from_rotvec(axis * angle)
        body_pose_param[counter] = torch.from_numpy(rot.as_rotvec()).type(torch.float32)
        # rot
        counter = counter + 1
        break
    body_pose_param = body_pose_param.view(-1).unsqueeze(0)

    return body_pose_param