# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torchvision import transforms
import kaolin as kal
from collections import defaultdict
import numpy as np
from kaolin.rep import TriangleMesh


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])


def get_pooling_index(positions, cam_mat, cam_pos, dims):
    # project points into 2D
    positions = positions * .57  # accounting for recaling in 3Dr2n
    positions = positions - cam_pos
    positions = torch.mm(positions, cam_mat.permute(1, 0))
    positions_xs = positions[:, 1] / positions[:, 2]
    positions_ys = -positions[:, 0] / positions[:, 2]

    # do bilinear interpolation over pixel coordiantes
    data_meta = defaultdict(list)


    for dim in dims:
        focal_length = 250. / 224. * dim
        xs = positions_xs * focal_length + dim / 2.
        ys = positions_ys * focal_length + dim / 2.

        cur_xs = torch.clamp(xs, 0, dim - 1)
        cur_ys = torch.clamp(ys, 0, dim - 1)

        x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)

        A = x2s - cur_xs
        B = cur_xs - x1s
        G = y2s - cur_ys
        H = cur_ys - y1s

        y1s = y1s + torch.arange(positions.shape[0]).float().to(positions.device) * dim
        y2s = y2s + torch.arange(positions.shape[0]).float().to(positions.device) * dim

        data_meta['A'].append(A.float().unsqueeze(0))
        data_meta['B'].append(B.float().unsqueeze(0))
        data_meta['G'].append(G.float().unsqueeze(0))
        data_meta['H'].append(H.float().unsqueeze(0))
        data_meta['x1s'].append(x1s.long().unsqueeze(0))
        data_meta['x2s'].append(x2s.long().unsqueeze(0))
        data_meta['y1s'].append(y1s.long().unsqueeze(0))
        data_meta['y2s'].append(y2s.long().unsqueeze(0))

    for key in data_meta:
        data_meta[key] = torch.cat(data_meta[key], dim=0)
    return data_meta


def pooling(blocks, pooling_indices):
    full_features = None
    for i_block, block in enumerate(blocks):
        A = pooling_indices['A'][i_block]
        B = pooling_indices['B'][i_block]
        G = pooling_indices['G'][i_block]
        H = pooling_indices['H'][i_block]

        x1s = pooling_indices['x1s'][i_block]
        x2s = pooling_indices['x2s'][i_block]
        y1s = pooling_indices['y1s'][i_block]
        y2s = pooling_indices['y2s'][i_block]


        C = torch.index_select(block, 1, x1s).view(block.shape[0], -1)
        C = torch.index_select(C, 1, y1s)
        D = torch.index_select(block, 1, x1s).view(block.shape[0], -1)
        D = torch.index_select(D, 1, y2s)
        E = torch.index_select(block, 1, x2s).view(block.shape[0], -1)
        E = torch.index_select(E, 1, y1s)
        F = torch.index_select(block, 1, x2s).view(block.shape[0], -1)
        F = torch.index_select(F, 1, y2s)


        features = (A * C * G + H * D * A + G * E * B + B * F * H).permute(1, 0)

        full_features = features if full_features is None else torch.cat((full_features, features), dim=1)

    return full_features

norm_distance = kal.metrics.point.SidedDistance()

def chamfer_normal(pred_mesh, gt_points, gt_norms):
    # find closest gt points
    gt_indices = norm_distance(pred_mesh.vertices.unsqueeze(0), gt_points.unsqueeze(0))[0]
    # select norms from closest points and exand to match edges lengths

    gt_norm_selections = gt_norms[gt_indices]
    new_dimensions = (gt_norm_selections.shape[0], pred_mesh.ve.shape[1], 3)
    vertex_norms = gt_norm_selections.view(-1, 1, 3).expand(new_dimensions)

    # get all neighbor positions
    neighbor_indecies = pred_mesh.vv.clone()
    empty_indecies = (neighbor_indecies >= 0)
    other_indecies = (neighbor_indecies < 0)
    neighbor_indecies[other_indecies] = 0
    empty_indecies = (empty_indecies).float().unsqueeze(-1)
    neighbor_indecies = neighbor_indecies.view(-1)
    vertex_neighbors = pred_mesh.vertices[neighbor_indecies].view(new_dimensions)

    # mask both tensors
    vertex_norms = vertex_norms * empty_indecies
    vertex_norms = vertex_norms.contiguous().view(-1, 3)
    vertex_neighbors = vertex_neighbors * empty_indecies
    vertex_neighbors = vertex_neighbors.contiguous().view(-1, 3)

    # calculate normal loss, devide by number of unmasked elements to get mean
    normal_loss = (torch.abs(torch.sum(vertex_norms * vertex_neighbors, dim=1)))
    normal_loss = normal_loss.sum() / float(empty_indecies.sum())
    return normal_loss




def setup_meshes(filename='meshes/156.obj', device="cuda"):
    mesh_1 = kal.rep.TriangleMesh.from_obj(filename, enable_adjacency=True)
    if device == 'cuda':
        mesh_1.cuda()
    adj_1 = mesh_1.compute_adjacency_matrix_full().clone()
    adj_1 = normalize_adj(adj_1)
    mesh_1_i = kal.rep.TriangleMesh.from_tensors(mesh_1.vertices.clone(), mesh_1.faces.clone())

    mesh_2, split_mx_1 = split_mesh(mesh_1)
    adj_2 = mesh_2.compute_adjacency_matrix_full().clone()
    adj_2 = normalize_adj(adj_2)
    mesh_2_i = kal.rep.TriangleMesh.from_tensors(mesh_2.vertices.clone(), mesh_2.faces.clone())

    mesh_3, split_mx_2 = split_mesh(mesh_2)
    adj_3 = mesh_3.compute_adjacency_matrix_full().clone()
    adj_3 = normalize_adj(adj_3)
    mesh_3_i = kal.rep.TriangleMesh.from_tensors(mesh_3.vertices.clone(), mesh_3.faces.clone())

    initial_meshes = [mesh_1_i, mesh_2_i, mesh_3_i]
    updated_meshes = [mesh_1, mesh_2, mesh_3]

    adjs = [adj_1, adj_2, adj_3]
    split_mxs = [split_mx_1, split_mx_2]
    mesh_info = {'init': initial_meshes, 'update': updated_meshes , 'adjs': adjs, 'split_mxs': split_mxs}

    return mesh_info


def normalize_adj(mx):
    rowsum = mx.sum(dim=1).view(-1)
    r_inv = 1. / rowsum
    r_inv[r_inv != r_inv] = 0.
    r_mat_inv = torch.eye(r_inv.shape[0]).to(mx.device) * r_inv
    mx = torch.mm(r_mat_inv, mx)
    return mx


def split(meshes, features, index):
    meshes['init'][index + 1].vertices = split_features(meshes['split_mxs'][index], meshes['update'][index].vertices)
    new_features = split_features(meshes['split_mxs'][index], features)
    return new_features


def split_mesh(mesh):
    faces = mesh.faces.clone()
    tracker = dict()
    vertex_count = mesh.vertices.shape[0]
    constant_vertex_count = vertex_count
    columns = np.zeros((vertex_count, 0))
    new_faces = []

    for face in faces:
        x, y, z = face.int()
        new_verts = []
        edges = [[x, y], [y, z], [z, x]]

        for a, b in edges:
            key = [a, b]
            key.sort()
            key = str(key)
            if key in tracker:
                new_verts.append(tracker[key])
            else:
                new_verts.append(vertex_count)
                column = np.zeros((constant_vertex_count, 1))
                column[a] = .5
                column[b] = .5
                columns = np.concatenate((columns, column), axis=1)
                tracker[key] = vertex_count
                vertex_count += 1

        v1, v2, v3 = new_verts
        new_faces.append([x, v1, v3])
        new_faces.append([v1, y, v2])
        new_faces.append([v2, z, v3])
        new_faces.append([v1, v2, v3])

    split_mx = torch.FloatTensor(columns).to(face.device)

    new_faces = torch.LongTensor(new_faces).to(face.device)

    new_verts = split_features(split_mx, mesh.vertices)
    updated_mesh = TriangleMesh.from_tensors(new_verts, new_faces, enable_adjacency=True)

    return updated_mesh, split_mx

def split_features(split_mx, features):
    features = features.permute(1, 0)
    new_features = torch.mm(features, split_mx)
    features = torch.cat((features, new_features), dim=1).permute(1, 0)
    return features

def loss_surf(meshes, tgt_points):
    loss = kal.metrics.point.chamfer_distance(meshes['update'][0].vertices, tgt_points, w1=1., w2=0.55)
    loss += kal.metrics.point.chamfer_distance(meshes['update'][1].vertices, tgt_points, w1=1., w2=0.55)
    loss += kal.metrics.point.chamfer_distance(meshes['update'][2].vertices, tgt_points, w1=1., w2=0.55)
    return loss

def loss_edge(meshes):
    loss = kal.metrics.mesh.edge_length(meshes['update'][0])
    loss += kal.metrics.mesh.edge_length(meshes['update'][1])
    loss += kal.metrics.mesh.edge_length(meshes['update'][2])
    return loss

def loss_lap(meshes):
    loss = .1 * kal.metrics.mesh.laplacian_loss(meshes['init'][0], meshes['update'][0])
    loss += kal.metrics.mesh.laplacian_loss(meshes['init'][1], meshes['update'][1])
    loss += kal.metrics.mesh.laplacian_loss(meshes['init'][2], meshes['update'][2])

    loss += torch.sum((meshes['init'][0].vertices - meshes['update'][0].vertices) ** 2, 1).mean() * .0666 * .1
    loss += torch.sum((meshes['init'][1].vertices - meshes['update'][1].vertices) ** 2, 1).mean() * .0666
    loss += torch.sum((meshes['init'][2].vertices - meshes['update'][2].vertices) ** 2, 1).mean() * .0666

    return loss

def loss_norm(meshes, tgt_points, tgt_norms):
    loss = chamfer_normal(meshes['update'][0], tgt_points, tgt_norms)
    loss += chamfer_normal(meshes['update'][1], tgt_points, tgt_norms)
    loss += chamfer_normal(meshes['update'][2], tgt_points, tgt_norms)
    return loss
