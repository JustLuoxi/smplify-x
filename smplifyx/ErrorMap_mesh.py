# load mesh
from plyfile import PlyData, PlyElement
import torch
import numpy as np

def WriteHandmesh(verts, faces, outmesh_path, color=None):
    outmesh_path=str(outmesh_path)
    verts = verts.cpu().numpy()
    if verts.shape[0] == 1:
        verts = verts.squeeze(0)
    with open(outmesh_path+".obj", 'w') as fp:
        i = 0
        for v in verts:
            fp.write('v %f %f %f' % (v[0], v[1], v[2]))
            if color is not None:
                fp.write(' %f %f %f\n' %(color[i][0],color[i][1],color[i][2]))
                i=i+1
            else:
                fp.write('\n')
        if faces is not None:
            faces = faces.cpu().numpy()
            for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print('..Output mesh saved to: ', outmesh_path)

def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    face = None
    
    faces = data['face']['vertex_indices']
    if faces.shape[0] != 0:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)
        face = face.transpose(1,0)

    data = {'pos':pos, 'face':face}

    return data

def main(src_path, tar_path, out_path):

    scr_data = read_ply(src_path)
    tar_data = read_ply(tar_path)

    # set map
    dist_mat = torch.cdist(scr_data['pos'], tar_data['pos'])
    min_dist = torch.min(dist_mat, dim=1)
    min_dist = min_dist[0]
    mean_dist = torch.mean(min_dist)
    print(out_path + ' MEAN: ' +  str(mean_dist))
    min_dist[min_dist>=0.08] = 0.08
    min = torch.min(min_dist)
    max = 0.08

    # save mesh
    dis_map = 1- (min_dist.unsqueeze(1) - min) / (max - min)
    import cv2
    dis_map =  dis_map.cpu().numpy()
    dis_map = np.array(dis_map * 255, dtype = np.uint8)
    color_map = cv2.applyColorMap(dis_map, cv2.COLORMAP_JET)
    color_map = np.array(color_map).squeeze()

    WriteHandmesh( scr_data['pos'], scr_data['face'], out_path, color=color_map)


root_folder = [
# 'X:\\POSE_EG\\0111_ExperimentData\\D_bouncing\\OURS\\70', 
# 'X:\\POSE_EG\\0111_ExperimentData\\I_crane\\OURS\\140', 
'X:\\POSE_EG\\0111_ExperimentData\\D_handstan\\OURS\\50', 
# 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\51', 
# 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\90',
# 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\120',
# 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\147',
# 'X:\\POSE_EG\\0111_ExperimentData\\T_samba\\OURS\\108',
]

import  os, subprocess
for root in root_folder:
    pathf = os.path.dirname(root)
    frame = root[len(pathf) + 1:]

    # ## clean surfacenet pc error map:   
    # # 1. mask clean pc:
    # exe_path = "X:\\Just\\OneDrive\\0_MyProjects\\POSE\\CODE\\4_pointcloud\\MaskCleanPC\\bin\\x64\\Release\\PVMap_Mesh.exe"
    # calib_folder = root + "\\Calib"
    # mask_folder = root + "\\mask"
    # pc_ply = root + "\\surfacenet\\all.ply"
    # clean_ply = root + "\\surfacenet\\maskclean_all.ply"
    # cmd = exe_path + ' '+ calib_folder+  ' ' + mask_folder + ' '+ pc_ply + ' ' + clean_ply
    # print(cmd)
    # subprocess.call(cmd)
    # print('$$$$$$$$ ' + clean_ply + ' is done' + '$$$$$$$$$$')

    # src_path1 = clean_ply
    src_path2 = root + "\\5_nonrigid\\" + 'nonrigidRefinded.ply'
    tar_path = root + '\\mesh_' + frame.zfill(6) + '.ply'
    # out_path1 = root + "\\surfacenet\\maskclean_all_errormap"
    out_path2 = root + "\\5_nonrigid\\" + frame.zfill(6) + '_errormap'
    # main(src_path1, tar_path, out_path1)
    main(src_path2,tar_path,out_path2)

    # ## clean PMVS pc error map:   
    # # 1. mask clean pc:
    # exe_path = "X:\\Just\\OneDrive\\0_MyProjects\\POSE\\CODE\\4_pointcloud\\MaskCleanPC\\bin\\x64\\Release\\PVMap_Mesh.exe"
    # calib_folder = root + "\\Calib"
    # mask_folder = root + "\\mask"
    # pc_ply = root + "\\pmvs_data\\models\\option-highres.txt.ply"
    # clean_ply = root + "\\pmvs_data\\models\\maskclean_all.ply"
    # cmd = exe_path + ' '+ calib_folder+  ' ' + mask_folder + ' '+ pc_ply + ' ' + clean_ply
    # print(cmd) 
    # subprocess.call(cmd) 
    # print('$$$$$$$$ ' + clean_ply + ' is done' + '$$$$$$$$$$') 

    # src_path1 = clean_ply
    # # src_path2 = root + "\\smplx_fitted_nonrigid\\" + frame.zfill(6) + '.ply'
    # tar_path = root + '\\mesh_' + frame.zfill(6) + '.ply'
    # out_path1 = root + "\\pmvs_data\\models\\maskclean_all_errormap"
    # # out_path2 = root + "\\smplx_fitted_nonrigid\\" + frame.zfill(6) + '_errormap'
    # main(src_path1, tar_path, out_path1)
    # # main(src_path2,tar_path,out_path2)

    # # visual hull:
    # src_path1 = root + "\\mask\\" + frame.zfill(6) + '.ply'
    # tar_path = root + '\\mesh_' + frame.zfill(6) + '.ply'
    # out_path1 = root + "\\mask\\" + frame.zfill(6) + '_errormap'
    # main(src_path1, tar_path, out_path1)