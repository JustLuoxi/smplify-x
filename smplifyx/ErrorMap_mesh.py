# load mesh
from plyfile import PlyData, PlyElement
import torch
import numpy as np

def WriteHandmesh(verts, faces, outmesh_path, color=None):
    outmesh_path=str(outmesh_path)
    verts = verts.cpu().numpy()
    if verts.shape[0] == 1:
        verts = verts.squeeze(0)
    faces = faces.cpu().numpy()
    with open(outmesh_path+".obj", 'w') as fp:
        i = 0
        for v in verts:
            fp.write('v %f %f %f' % (v[0], v[1], v[2]))
            if color is not None:
                fp.write(' %f %f %f\n' %(color[i][0],color[i][1],color[i][2]))
                i=i+1
            else:
                fp.write('\n')

        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    print('..Output mesh saved to: ', outmesh_path)

def read_ply(path):
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = {'pos':pos, 'face':face.transpose(1,0)}

    return data

src_path = 'X:\\POSE_EG\\0111_ExperimentData\\D_bouncing\\OURS\\70\\smplx_fitted\\000070.ply'
tar_path = 'X:\\POSE_EG\\0111_ExperimentData\\D_bouncing\\OURS\\70\\mesh_000070.ply'
# src_ply = PlyData.read(src_path)
# tar_ply = PlyData.read(tar_path)
#
# # compute distance
# pos_src = ([torch.tensor(src_ply['vertex'][axis]) for axis in ['x', 'y', 'z']])
# pos_src = torch.stack(pos_src, dim=-1)
# pos_tar = ([torch.tensor(tar_ply['vertex'][axis]) for axis in ['x', 'y', 'z']])
# pos_tar = torch.stack(pos_tar, dim=-1)

scr_data = read_ply(src_path)
tar_data = read_ply(tar_path)

# set map
dist_mat = torch.cdist(scr_data['pos'], tar_data['pos'])
min_dist = torch.min(dist_mat, dim=1)
min_dist = min_dist[0]
min_dist[min_dist>=0.08] = 0.08
mean_dist = torch.mean(min_dist)
print(mean_dist)
min = torch.min(min_dist)
max = 0.08

# save mesh
dis_map = 1- (min_dist.unsqueeze(1) - min) / (max - min)
import cv2
dis_map =  dis_map.cpu().numpy()
dis_map = np.array(dis_map * 255, dtype = np.uint8)
color_map = cv2.applyColorMap(dis_map, cv2.COLORMAP_JET)
color_map = np.array(color_map).squeeze()

out_path = 'X:\\POSE_EG\\0111_ExperimentData\\D_bouncing\\OURS\\70\\smplx_fitted\\000070_errormap'
WriteHandmesh( scr_data['pos'], scr_data['face'], out_path, color=color_map)