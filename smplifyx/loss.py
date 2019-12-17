import torch
import time
import numpy as np
import torch.nn.functional as torch_f
from alignment.utils import AnalyzeSkeleton
import math
import torch.nn.functional as F

def ShapeRegLoss(shape_parameter, lmb=1.0):
    shape_reg_loss = torch_f.mse_loss(
        shape_parameter, torch.zeros_like(shape_parameter), reduction='mean'
    )
    return shape_reg_loss*lmb


def TemporalLoss(pose_parameter, last_frame_pose):
    loss_global = torch_f.mse_loss(pose_parameter[:, 0:3], last_frame_pose[:, 0:3], reduction='mean')
    loss_pose = torch_f.mse_loss(pose_parameter[:, 3:], last_frame_pose[:, 3:], reduction='mean')

    return 0.6*loss_global + 0.4*loss_pose


def PoseRegLoss(pose_parameter, last_frame_pose=None):
    if last_frame_pose is not None:
        return TemporalLoss(pose_parameter, last_frame_pose)
    loss = torch_f.mse_loss(pose_parameter, torch.zeros_like(pose_parameter), reduction='mean')

    return loss

def PoseRegManoLoss(pose_parameter, original_pose_params):
    loss = torch_f.mse_loss(pose_parameter, original_pose_params, reduction='mean')

    return loss


def PoseRegPCALoss(pose_parameter,pc_param_norm, pc_param_std, w_norm = 1, bone_id = None):

    if bone_id == None:
        loss1 = torch_f.mse_loss(pose_parameter, pc_param_norm, reduction='mean')
        loss2 = F.relu(torch.abs(pose_parameter)-pc_param_std)
        loss2 = torch_f.mse_loss(loss2, torch.zeros_like(loss2), reduction='mean')
    else:
        a = pose_parameter[:,3*(bone_id-1):3*bone_id].clone()
        b = pc_param_norm[:,3*(bone_id-1):3*bone_id].clone()
        c = pc_param_std[3 * (bone_id - 1):3 * bone_id].clone()
        loss1 = torch_f.mse_loss(a, b)
        loss2 = F.relu(torch.abs(a)-c)
        loss2 = torch_f.mse_loss(loss2, torch.zeros_like(loss2))

    loss = (w_norm*loss1 + loss2)

    return loss

def PoseRegPCALoss_Level(pose_parameter,pc_param_norm, pc_param_std, w_norm = 1, level_ids = None):

    if level_ids == None:
        loss1 = torch_f.mse_loss(pose_parameter, pc_param_norm, reduction='mean')
        loss2 = F.relu(torch.abs(pose_parameter)-pc_param_std)
        loss2 = torch_f.mse_loss(loss2, torch.zeros_like(loss2), reduction='mean')
    else:
        loss1 = torch.tensor(0).cuda().float()
        loss2 = torch.tensor(0).cuda().float()
        for bone_id in level_ids:
            a = pose_parameter[:,3*(bone_id-1):3*bone_id].clone()
            b = pc_param_norm[:,3*(bone_id-1):3*bone_id].clone()
            c = pc_param_std[3*(bone_id-1):3*bone_id].clone()

            loss1 += torch_f.mse_loss(a, b)
            loss_t = F.relu(torch.abs(a)-c)
            loss2 += torch_f.mse_loss(loss_t, torch.zeros_like(loss_t))

    loss = (w_norm*loss1 + loss2)

    return loss

def JointPositionLoss(pred_verts, target_verts, handsize=1): #, skipjoints=[1,9,12,23,24,20,21]):

    standard = handsize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # pred_verts[:,skipjoints,:] = 0
    # target_verts[:,skipjoints,:] = 0
    loss = loss_fn(pred_verts[:, 1:, :], target_verts[:, 1:, :])
    loss = loss / (standard + 1e-8)
    return loss

def JointPositionLoss_Just(pred_joints, verts, target_joints, handsize=1): #, skipjoints=[1,9,12,23,24,20,21]):

    p_idxs = torch.LongTensor([0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]).cuda(0)
    t_idxs = torch.LongTensor([0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]).cuda(0)

    p_id_2 = torch.LongTensor([125,1113,1530,472,825]).cuda(0)
    t_id_2 = torch.LongTensor([4,8,12,16,20]).cuda(0)

    standard = handsize
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # pred_verts[:,skipjoints,:] = 0
    # target_verts[:,skipjoints,:] = 0
    pp = torch.index_select(pred_joints,1,p_idxs)
    tt = torch.index_select(target_joints,1,t_idxs)
    loss1 = loss_fn(pp, tt)
    loss1 = loss1 / (standard + 1e-8)
    pp = torch.index_select(verts, 1, p_id_2)
    tt = torch.index_select(target_joints,1,t_id_2)
    loss2 = loss_fn(pp, tt)
    loss2 = loss2 / (standard + 1e-8)
    return loss1 + loss2


def BoneLengthLoss(pred_verts, target_verts, isBody=False):
    assert (pred_verts.shape == target_verts.shape)
    batch_size = pred_verts.shape[0]
    if isBody:
        links = [
            (0,1,8),
            (2,3,4),
            (5,6,7),
            (10,11),
            (13,14),
            (2,1,5),
        ]
        bone_num = 10
    else:
        links = torch.tensor([
            (0, 1, 2, 3, 4),
            (0, 5, 6, 7, 8),
            (0, 9, 10, 11, 12),
            (0, 13, 14, 15, 16),
            (0, 17, 18, 19, 20),
        ]).cuda()
        bone_num = 20

    bone_ratios = torch.zeros((batch_size, bone_num)).cuda()
    ones = torch.ones_like(bone_ratios).cuda()
    itr = 0
    for link in links:
        for j1, j2 in zip(link[0:-1], link[1:]):
            pred_bone_len = torch.norm(pred_verts[:, j1, :] - pred_verts[:, j2, :], dim=1)
            target_bone_len = torch.norm(target_verts[:, j1, :] - target_verts[:, j2, :], dim=1)
            ratio = pred_bone_len / target_bone_len
            bone_ratios[:, itr] = ratio
            itr = itr+1
    loss = torch_f.mse_loss(bone_ratios, ones, reduction='mean')
    return loss * 50

def JointAngleLoss(pred_verts, tar_angles, tar_bone_dirs):
    pred_angles, pred_bone_dirs = AnalyzeSkeleton(pred_verts)

    # angle: 0-2pi
    loss_angle = torch_f.mse_loss(pred_angles, tar_angles, reduction='mean')

    # # angle:0-2pi
    # pred_dir_angles = torch.acos((pred_bone_dirs * tar_bone_dirs).sum(dim=2))
    # optim_dir_angles = torch.zeros_like(pred_dir_angles)
    # loss_dir = torch_f.mse_loss(pred_dir_angles, optim_dir_angles, reduction='mean')

    loss = loss_angle*1000  #+ loss_dir

    return loss

def JointAngleLoss_Just(pred_verts, verts, target_skeleton, tar_angles, tar_bone_dirs):

    pred_verts_fake = pred_verts.clone()
    pred_verts_fake[:,4,:] = verts[:,125,:]
    pred_verts_fake[:,8,:] = verts[:,1113,:]
    pred_verts_fake[:,12,:] = verts[:,1530,:]
    pred_verts_fake[:,16,:] = verts[:,472,:]
    t = verts[:,825,:].unsqueeze(1)
    pred_verts_fake = torch.cat([pred_verts_fake, t], 1)
    # t = target_skeleton[:,20,:].unsqueeze(1)
    # pred_verts_fake = torch.cat([pred_verts_fake, t], 1)
    pred_angles, pred_bone_dirs = AnalyzeSkeleton(pred_verts_fake)

    # angle: 0-2pi
    loss_angle = torch_f.mse_loss(pred_angles, tar_angles, reduction='mean')
    # select_idx = [0,1,2,5,6,9,10,13,14,17,18]

    # angle:0-2pi
    # pred_dir_angles = torch.acos((pred_bone_dirs * tar_bone_dirs).sum(dim=2))
    # optim_dir_angles = torch.zeros_like(pred_dir_angles)
    # loss_dir = torch_f.mse_loss(pred_dir_angles, optim_dir_angles, reduction='mean')

    loss = loss_angle*1000

    return loss

def PtcloudLoss(pred_mesh_verts, target_pt_cloud, target_normal, selected_indices=None, handsize=1):
    # 1*3817*3
    # 1*N*3
    # flag_selected: True: only use the selected verts as regression loss, False: all

    if selected_indices is not None:
        pred_mesh_verts = torch.index_select(pred_mesh_verts,1,selected_indices)

    target_pt_cloud = target_pt_cloud.squeeze(0)
    target_normal = target_normal.squeeze(0)
    pred_mesh_verts = pred_mesh_verts.squeeze(0)
    pt_cloud = target_pt_cloud[:, 0:3]
    pt_cloud_normal = torch.nn.functional.normalize(target_normal)
    dist_mat = torch.cdist(pred_mesh_verts, pt_cloud)
    # print(dist_mat)
    minimum = torch.min(dist_mat, dim=1)

    # print(minimum)
    # dist_mat = torch.norm(pred_mesh_verts - pt_cloud, dim=1)
    # return torch.mean(dist_mat)

    match_pt = target_pt_cloud[minimum.indices]
    # np.savetxt("match_pt.xyz", match_pt.cpu().detach(), fmt="%.6f")
    x = pt_cloud[minimum.indices,:]
    n = pt_cloud_normal[minimum.indices,:]
    p2pl = torch.sum((pred_mesh_verts - x)*n, dim=1)
    p2pl = p2pl / (handsize + 1e-8)
    p2pl_loss = torch_f.mse_loss(p2pl, torch.zeros_like(p2pl))

    # distances = minimum.values
    # p2p_loss = torch_f.mse_loss(distances, torch.zeros_like(distances))



    return p2pl_loss

# find all correspondences of [pre] in [target]
def FindCorrespondence(pre_verts, target_pt_cloud, pre_normals, target_normals, K=1):

    # l2 distance
    thre_l2 = 2 # cm
    dist_mat = torch.cdist(pre_verts, target_pt_cloud)
    E_dist_mat = torch.exp(10/thre_l2*dist_mat)

    # cos distance
    thre_cos = math.cos(math.pi/6) #
    f_pre = F.normalize(pre_normals)
    f_target = F.normalize(target_normals)
    cosd_mat = f_pre.mm(f_target.t())
    E_cosd_mat = torch.exp(10/thre_cos*(1-cosd_mat))

    a = 0.5
    b = 1-a

    D_mat = a*E_dist_mat + b* E_cosd_mat
    values, indices = torch.topk(D_mat, K, dim=1,largest=False)
    # minimal = torch.min(D_mat, dim=1)
    # a = torch.tensor(10,dtype=torch.float32, device='cuda' )
    # values = minimal[0]
    # indices = minimal[1]

    return values, indices

# add semantic to help find correspondence
def FindCorrespondence_Label(pre_verts, target_pt_cloud, pre_normals, target_normals, pre_labels, target_labels, K=1):

    # l2 distance
    thre_l2 = 2 # cm
    dist_mat = torch.cdist(pre_verts, target_pt_cloud)
    E_dist_mat = torch.exp(10/thre_l2*dist_mat)

    # cos distance
    thre_cos = math.cos(math.pi/6) #
    f_pre = F.normalize(pre_normals)
    f_target = F.normalize(target_normals)
    cosd_mat = f_pre.mm(f_target.t())
    E_cosd_mat = torch.exp(10/thre_cos*(1-cosd_mat))

    a = 0.5
    b = 1-a

    # semantic distance
    dis_sem_mat = pre_labels.float().mm((1/target_labels.float()).t())
    flag = torch.eq(dis_sem_mat, 1)
    dis_sem_mat[flag] = 0
    flag = torch.gt(dis_sem_mat, 0)
    dis_sem_mat[flag] = 1e30

    D_mat = a*E_dist_mat + b* E_cosd_mat + dis_sem_mat
    values, indices = torch.topk(D_mat, K, dim=1,largest=False)
    # minimal = torch.min(D_mat, dim=1)
    # a = torch.tensor(10,dtype=torch.float32, device='cuda' )
    # values = minimal[0]
    # indices = minimal[1]

    return values, indices

# TODO: 1113 increase selected vertices weights
def PositionNormalJointLoss_Just(pre_verts, target_pt_cloud, pre_normals, target_normals, selected_indices=None):

    v_w = torch.ones([pre_verts.shape[1]]).cuda()

    if selected_indices is not None:
        # pre_verts = torch.index_select(pre_verts,1,selected_indices)
        # pre_normals = torch.index_select(pre_normals, 1, selected_indices)
        v_w[selected_indices] = pre_verts.shape[1]/len(selected_indices)

    target_pt_cloud = target_pt_cloud.squeeze(0)
    pre_verts = pre_verts.squeeze(0)
    target_normals = target_normals.squeeze(0)
    pre_normals = pre_normals.squeeze(0)

    K=1
    # vert -> pt
    values, idx_vnpt = FindCorrespondence(pre_verts, target_pt_cloud, pre_normals, target_normals, K=K)
    if len(values.shape)==1:
        values.unsqueeze_(1)
        idx_vnpt.unsqueeze_(1)
    # w = F.normalize(values,p=1)
    x = target_pt_cloud[idx_vnpt[:]]
    verts = pre_verts.unsqueeze(1)
    verts = verts.repeat(1,K,1)
    dis = torch.norm(verts - x, 2, 2)
    # dis = torch.sum(w*dis,1)
    new_w = v_w.unsqueeze(1)
    new_w = new_w.repeat(1,K)
    dis = dis * new_w
    l1 = torch.mean(dis)

    # x = target_pt_cloud[idx_vnpt, :]
    # l1 = torch.mean(torch.norm(pre_verts - x, 2,1))

    # if selected_indices is None:
    # pt -> vert
    values1, idx_ptnvert = FindCorrespondence(target_pt_cloud,pre_verts, target_normals, pre_normals, K=K)
    if len(values1.shape)==1:
        values1.unsqueeze_(1)
        idx_ptnvert.unsqueeze_(1)
    # w1 = F.normalize(values1,p=1)
    x1 = pre_verts[idx_ptnvert]
    verts1 = target_pt_cloud.unsqueeze(1)
    verts1 = verts1.repeat(1, K, 1)
    dis1 = torch.norm(verts1 - x1, 2, 2)
    # dis1 = torch.sum(w1 * dis1, 1)
    new_w = v_w[idx_ptnvert]
    dis1 = dis1 * new_w
    l2 = torch.mean(dis1)
    l1 += l2

    # x = pre_verts[idx_ptnvert, :]
    # l2 = torch.mean(torch.norm(target_pt_cloud - x, 2,1))

    # # bidirection nearest
    # i = torch.range(0, len(idx_vnpt)-1).cuda(0).long()
    # flag = torch.eq(idx_ptnvert[idx_vnpt[i]], i)
    # flag = torch.nonzero(flag).squeeze()
    # x = pre_verts[flag,:]
    # y = target_pt_cloud[idx_vnpt[flag],:]
    # # distance loss
    # l1 = torch.mean(torch.norm(y - x, 2,1))

    # # # TODO: find the overlap loss
    # a = torch.tensor(pre_verts.shape[0] / len(flag),  dtype=torch.float32, device='cuda')
    # offset = torch.tensor(1, dtype=torch.float32, device='cuda')
    # l2 = torch.exp(a) - torch.exp(offset)

    loss = l1

    # TODO: Geman-McClure

    return loss

def PositionNormalLabelJointLoss_Just(pre_verts, target_pt_cloud, pre_normals, target_normals, pre_labels, target_labels, selected_indices=None):

    if selected_indices is not None:
        pre_verts = torch.index_select(pre_verts,1,selected_indices)
        pre_normals = torch.index_select(pre_normals, 1, selected_indices)
        pre_labels = pre_labels[selected_indices]

    target_pt_cloud = target_pt_cloud.squeeze(0)
    pre_verts = pre_verts.squeeze(0)
    target_normals = target_normals.squeeze(0)
    pre_normals = pre_normals.squeeze(0)

    K=1
    # vert -> pt
    values, idx_vnpt = FindCorrespondence_Label(pre_verts, target_pt_cloud, pre_normals, target_normals,pre_labels,target_labels, K=K)
    if len(values.shape)==1:
        values.unsqueeze_(1)
        idx_vnpt.unsqueeze_(1)
    # w = F.normalize(values,p=1)
    x = target_pt_cloud[idx_vnpt[:]]
    verts = pre_verts.unsqueeze(1)
    verts = verts.repeat(1,K,1)
    dis = torch.norm(verts - x, 2, 2)
    # dis = torch.sum(w*dis,1)
    l1 = torch.mean(dis)

    # x = target_pt_cloud[idx_vnpt, :]
    # l1 = torch.mean(torch.norm(pre_verts - x, 2,1))

    # pt -> vert
    values1, idx_ptnvert = FindCorrespondence_Label(target_pt_cloud,pre_verts, target_normals, pre_normals,target_labels,pre_labels, K=K)
    if len(values1.shape)==1:
        values1.unsqueeze_(1)
        idx_ptnvert.unsqueeze_(1)
    # w1 = F.normalize(values1,p=1)
    x1 = pre_verts[idx_ptnvert]
    verts1 = target_pt_cloud.unsqueeze(1)
    verts1 = verts1.repeat(1, K, 1)
    dis1 = torch.norm(verts1 - x1, 2, 2)
    # dis1 = torch.sum(w1 * dis1, 1)
    l2 = torch.mean(dis1)

    # x = pre_verts[idx_ptnvert, :]
    # l2 = torch.mean(torch.norm(target_pt_cloud - x, 2,1))

    loss = l1 + l2
    # torch_f.mse_loss(loss, torch.zeros_like(loss))
    return loss

# add hierarchy weights
def PositionNormalJointLoss_Just_w(pre_verts, target_pt_cloud, pre_normals, target_normals, v_w, selected_indices=None):

    if selected_indices is not None:
        pre_verts = torch.index_select(pre_verts,1,selected_indices)
        pre_normals = torch.index_select(pre_normals, 1, selected_indices)
        v_w = v_w[selected_indices]

    target_pt_cloud = target_pt_cloud.squeeze(0)
    pre_verts = pre_verts.squeeze(0)
    target_normals = target_normals.squeeze(0)
    pre_normals = pre_normals.squeeze(0)

    K=1
    # vert -> pt
    values, idx_vnpt = FindCorrespondence(pre_verts, target_pt_cloud, pre_normals, target_normals, K=K)
    if len(values.shape)==1:
        values.unsqueeze_(1)
        idx_vnpt.unsqueeze_(1)
    # w = F.normalize(values,p=1)
    x = target_pt_cloud[idx_vnpt[:]]
    verts = pre_verts.unsqueeze(1)
    verts = verts.repeat(1,K,1)
    dis = torch.norm(verts - x, 2, 2)
    # dis = torch.sum(w*dis,1)
    new_w = v_w.unsqueeze(1)
    new_w = new_w.repeat(1,K)
    dis = dis * new_w
    l1 = torch.mean(dis)

    # x = target_pt_cloud[idx_vnpt, :]
    # l1 = torch.mean(torch.norm(pre_verts - x, 2,1))

    # pt -> vert
    values1, idx_ptnvert = FindCorrespondence(target_pt_cloud,pre_verts, target_normals, pre_normals, K=K)
    if len(values1.shape)==1:
        values1.unsqueeze_(1)
        idx_ptnvert.unsqueeze_(1)
    # w1 = F.normalize(values1,p=1)
    x1 = pre_verts[idx_ptnvert]
    verts1 = target_pt_cloud.unsqueeze(1)
    verts1 = verts1.repeat(1, K, 1)
    dis1 = torch.norm(verts1 - x1, 2, 2)
    # dis1 = torch.sum(w1 * dis1, 1)
    new_w = v_w[idx_ptnvert]
    dis1 = dis1 * new_w
    l2 = torch.mean(dis1)

    # x = pre_verts[idx_ptnvert, :]
    # l2 = torch.mean(torch.norm(target_pt_cloud - x, 2,1))

    # # bidirection nearest
    # i = torch.range(0, len(idx_vnpt)-1).cuda(0).long()
    # flag = torch.eq(idx_ptnvert[idx_vnpt[i]], i)
    # flag = torch.nonzero(flag).squeeze()
    # x = pre_verts[flag,:]
    # y = target_pt_cloud[idx_vnpt[flag],:]
    # # distance loss
    # l1 = torch.mean(torch.norm(y - x, 2,1))

    # # # TODO: find the overlap loss
    # a = torch.tensor(pre_verts.shape[0] / len(flag),  dtype=torch.float32, device='cuda')
    # offset = torch.tensor(1, dtype=torch.float32, device='cuda')
    # l2 = torch.exp(a) - torch.exp(offset)


    loss = 2*l1 + 2*l2
    # torch_f.mse_loss(loss, torch.zeros_like(loss))
    return loss

def PtChamferLoss(pred_mesh_verts, target_pt_cloud, selected_indices=None):
    if selected_indices is not None:
        pred_mesh_verts = torch.index_select(pred_mesh_verts,1,selected_indices)

    target_pt_cloud = target_pt_cloud.squeeze(0)
    pred_mesh_verts = pred_mesh_verts.squeeze(0)
    pt_cloud = target_pt_cloud[:, :3]
    dist_mat1 = torch.cdist(pred_mesh_verts, pt_cloud)
    minimum1 = torch.min(dist_mat1, dim=1)
    x = pt_cloud[minimum1.indices, :]
    l1 = torch.mean(torch.norm(pred_mesh_verts - x, 2,1))

    dist_mat2 = torch.cdist(pt_cloud,pred_mesh_verts)
    minimum2 = torch.min(dist_mat2, dim=1)
    x = pred_mesh_verts[minimum2.indices,:]
    l2 = torch.mean(torch.norm(pt_cloud-x,2,1))

    loss = l1 + l2
    torch_f.mse_loss(loss, torch.zeros_like(loss))
    return loss


def BoneDirectionLoss(pred_verts, target_verts):
    raise NotImplementedError


if __name__ == "__main__":
    a = torch.rand(1,2,3).cuda()
    print(a)
    b = torch.rand(1,2,6).cuda()
    print(b)
    loss = PtcloudLoss(a, b)
    # print(loss)
    # l2l = JointPositionLoss(a, b)
    # print(l2l)
    # bonel = BoneLengthLoss(a, b)
    # print(bonel)
    # c, d = AnalyzeSkeleton(b)
    # angl = JointAngleLoss(a, c, d)
    # print(angl)