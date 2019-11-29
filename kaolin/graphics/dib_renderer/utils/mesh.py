# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import cv2
import torch
import numpy as np


##################################################################
# faces begin from 0!!!
def face2edge(facenp_fx3):
    '''
    facenp_fx3, int32
    return edgenp_ex2, int32
    '''
    f1 = facenp_fx3[:, 0:1]
    f2 = facenp_fx3[:, 1:2]
    f3 = facenp_fx3[:, 2:3]
    e1 = np.concatenate((f1, f1, f2), axis=0)
    e2 = np.concatenate((f2, f3, f3), axis=0)
    edgenp_ex2 = np.concatenate((e1, e2), axis=1)
    # sort & unique
    edgenp_ex2 = np.sort(edgenp_ex2, axis=1)
    edgenp_ex2 = np.unique(edgenp_ex2, axis=0)
    return edgenp_ex2


def face2edge2(facenp_fx3, edgenp_ex2):
    '''
    facenp_fx3, int32
    edgenp_ex2, int32
    return face_fx3, int32
    this face is indexed by edge
    '''
    fnum = facenp_fx3.shape[0]
    enum = edgenp_ex2.shape[0]

    edgesort = np.sort(edgenp_ex2, axis=1)
    edgere_fx3 = np.zeros_like(facenp_fx3)
    for i in range(fnum):
        for j in range(3):
            pbe, pen = facenp_fx3[i, j], facenp_fx3[i, (j + 1) % 3]
            if pbe > pen:
                pbe, pen = pen, pbe
            cond = (edgesort[:, 0] == pbe) & (edgesort[:, 1] == pen)
            idx = np.where(cond)[0]
            edgere_fx3[i, j] = idx
    return edgere_fx3


def edge2face(facenp_fx3, edgenp_ex2):
    '''
    facenp_fx3, int32
    edgenp_ex2, int32
    return edgenp_ex2, int32
    this edge is indexed by face
    '''
    fnum = facenp_fx3.shape[0]
    enum = edgenp_ex2.shape[0]

    facesort = np.sort(facenp_fx3, axis=1)
    edgesort = np.sort(edgenp_ex2, axis=1)
    edgere_ex2 = np.zeros_like(edgesort)
    for i in range(enum):
        pbe, pen = edgesort[i]
        eid = 0
        for j in range(fnum):
            f1, f2, f3 = facesort[j]
            cond1 = f1 == pbe and f2 == pen
            cond2 = f1 == pbe and f3 == pen
            cond3 = f2 == pbe and f3 == pen
            if cond1 or cond2 or cond3:
                edgere_ex2[i, eid] = j
                eid += 1

    return edgere_ex2


def face2pneimtx(facenp_fx3):
    '''
    facenp_fx3, int32
    return pointneighbourmtx, pxp, float32
    will normalize!
    assume it is a good mesh
    every point has more than one neighbour
    '''
    pnum = np.max(facenp_fx3) + 1
    pointneighbourmtx = np.zeros(shape=(pnum, pnum), dtype=np.float32)
    for i in range(3):
        be = i
        en = (i + 1) % 3
        idx1 = facenp_fx3[:, be]
        idx2 = facenp_fx3[:, en]
        pointneighbourmtx[idx1, idx2] = 1
        pointneighbourmtx[idx2, idx1] = 1
    pointneicount = np.sum(pointneighbourmtx, axis=1, keepdims=True)
    assert np.all(pointneicount > 0)
    pointneighbourmtx /= pointneicount
    return pointneighbourmtx


def face2pfmtx(facenp_fx3):
    '''
    facenp_fx3, int32
    reutrn pfmtx, pxf, float32
    '''
    pnum = np.max(facenp_fx3) + 1
    fnum = facenp_fx3.shape[0]
    pfmtx = np.zeros(shape=(pnum, fnum), dtype=np.float32)
    for i, f in enumerate(facenp_fx3):
        pfmtx[f[0], i] = 1
        pfmtx[f[1], i] = 1
        pfmtx[f[2], i] = 1
    return pfmtx


# upsample new points
def meshresample(pointnp_px3, facenp_fx3, edgenp_ex2):
    p1 = pointnp_px3[edgenp_ex2[:, 0], :]
    p2 = pointnp_px3[edgenp_ex2[:, 1], :]
    pmid = (p1 + p2) / 2
    point2np_px3 = np.concatenate((pointnp_px3, pmid), axis=0)

    # delete f
    # add 4 new faces
    face2np_fx3 = []
    pnum = np.max(facenp_fx3) + 1
    for f in facenp_fx3:
        p1, p2, p3 = f
        p12 = (edgenp_ex2 == (min(p1, p2), max(p1, p2))).all(axis=1).nonzero()[0] + pnum
        p23 = (edgenp_ex2 == (min(p2, p3), max(p2, p3))).all(axis=1).nonzero()[0] + pnum
        p31 = (edgenp_ex2 == (min(p3, p1), max(p3, p1))).all(axis=1).nonzero()[0] + pnum
        face2np_fx3.append([p1, p12, p31])
        face2np_fx3.append([p12, p2, p23])
        face2np_fx3.append([p31, p23, p3])
        face2np_fx3.append([p12, p23, p31])
    face2np_fx3 = np.array(face2np_fx3, dtype=np.int64)
    return point2np_px3, face2np_fx3


def mtx2tfsparse(mtx):
    m, n = mtx.shape
    rows, cols = np.nonzero(mtx)
    # N = rows.shape[0]
    # value = np.ones(shape=(N,), dtype=np.float32)
    value = mtx[rows, cols]
    v = torch.FloatTensor(value)
    i = torch.LongTensor(np.stack((rows, cols), axis=0))
    tfspmtx = torch.sparse.FloatTensor(i, v, torch.Size([m, n]))
    return tfspmtx


################################################################
def loadobj(meshfile):

    v = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            continue
        if data[0] == 'v':
            v.append([float(d) for d in data[1:]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    return pointnp_px3, facenp_fx3


def loadobjcolor(meshfile):

    v = []
    vc = []
    f = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if data[0] == 'v':
            v.append([float(d) for d in data[1:4]])
            if len(data) == 7:
                vc.append([float(d) for d in data[4:7]])
        if data[0] == 'f':
            data = [da.split('/')[0] for da in data]
            f.append([int(d) for d in data[1:4]])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    if len(vc) > 0:
        vc = np.array(vc, dtype=np.float32)
    else:
        vc = np.ones_like(pointnp_px3)
    return pointnp_px3, facenp_fx3, vc


def loadobjtex(meshfile):

    v = []
    vt = []
    f = []
    ft = []
    meshfp = open(meshfile, 'r')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if not ((len(data) == 3) or (len(data) == 4) or (len(data) == 5)):
            continue
        if data[0] == 'v':
            if len(data) == 4:
                v.append([float(d) for d in data[1:]])
        if data[0] == 'vt':
            if len(data) == 3 or len(data) == 4:
                vt.append([float(d) for d in data[1:3]])
        if data[0] == 'f':
            data = [da.split('/') for da in data]
            if len(data) == 4:
                f.append([int(d[0]) for d in data[1:]])
                ft.append([int(d[1]) for d in data[1:]])
            elif len(data) == 5:
                idx1 = [1, 2, 3]
                data1 = [data[i] for i in idx1]
                f.append([int(d[0]) for d in data1])
                ft.append([int(d[1]) for d in data1])
                idx2 = [1, 3, 4]
                data2 = [data[i] for i in idx2]
                f.append([int(d[0]) for d in data2])
                ft.append([int(d[1]) for d in data2])
    meshfp.close()

    # torch need int64
    facenp_fx3 = np.array(f, dtype=np.int64) - 1
    ftnp_fx3 = np.array(ft, dtype=np.int64) - 1
    pointnp_px3 = np.array(v, dtype=np.float32)
    uvs = np.array(vt, dtype=np.float32)
    return pointnp_px3, facenp_fx3, uvs, ftnp_fx3


def savemesh(pointnp_px3, facenp_fx3, fname, partinfo=None):

    if partinfo is None:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            if partinfo[pidx, -1] == 0:
                pp = p
                color = [1, 0, 0]
            else:
                pp = p
                color = [0, 0, 1]
            fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return


def savemeshcolor(pointnp_px3, facenp_fx3, fname, color_px3=None):

    if color_px3 is None:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    else:
        fid = open(fname, 'w')
        for pidx, p in enumerate(pointnp_px3):
            pp = p
            color = color_px3[pidx]
            fid.write('v %f %f %f %f %f %f\n' % (pp[0], pp[1], pp[2], color[0], color[1], color[2]))
        for f in facenp_fx3:
            f1 = f + 1
            fid.write('f %d %d %d\n' % (f1[0], f1[1], f1[2]))
        fid.close()
    return


def savemeshtes(pointnp_px3, tcoords_px2, facenp_fx3, fname):

    import os
    fol, na = os.path.split(fname)
    na, _ = os.path.splitext(na)

    matname = '%s/%s.mtl' % (fol, na)
    fid = open(matname, 'w')
    fid.write('newmtl material_0\n')
    fid.write('Kd 1 1 1\n')
    fid.write('Ka 0 0 0\n')
    fid.write('Ks 0.4 0.4 0.4\n')
    fid.write('Ns 10\n')
    fid.write('illum 2\n')
    fid.write('map_Kd %s.png\n' % na)
    fid.close()

    fid = open(fname, 'w')
    fid.write('mtllib %s.mtl\n' % na)

    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write('v %f %f %f\n' % (pp[0], pp[1], pp[2]))

    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write('vt %f %f\n' % (pp[0], pp[1]))

    fid.write('usemtl material_0\n')
    for f in facenp_fx3:
        f1 = f + 1
        fid.write('f %d/%d %d/%d %d/%d\n' % (f1[0], f1[0], f1[1], f1[1], f1[2], f1[2]))
    fid.close()

    return


def save_textured_mesh(directory,
                       file_name,
                       vertex_pos_px3,
                       face_fx3,
                       tex_coord_px2,
                       texture_hxwxc,
                       normalize_tex_coord=False,
                       flip_vertical=False,
                       texture_bias=0.01):
    """
    Save a textured mesh.

    Args:
        directory (str): The path to the folder containing the mesh to be saved.
        file_name (str): The name of the mesh to be saved (without extension).
            <file_name>.obj, <file_name>.mtl, and <file_name>.png will be saved.

        vertex_pos_px3 (numpy.ndarray): An array of shape (num_points, 3).
            Denotes the vertex position.

        face_fx3 (numpy.ndarray): An array of shape (num_faces, 3).
            Specifies, for each face, which vertices are used.

        tex_coord_px2 (numpy.ndarray): An array of shape (num_points, 2).
            Specifies the texture coordinate of each vertex.
            Each coordinate should be in the range [0, 1] or [-1, -1].
            If the range is [-1, -1], set normalize_tex_coord to True.

            NOTE: if this array is of the same format as specified for
            torch.nn.functional.grid_sample(), set both normalize_tex_coord
            and flip_vertical to True.

        texture_hxwxc (numpy.ndarray): An array of shape (height, width, channels).
            The texture to be saved. Each channel should be in the range [0, 1].

        normalize_tex_coord (bool): Whether to normalize texture coordinates,
            from [-1, 1] to [0, 1].

        flip_vertical (bool): Whether to flip the texture coordinates vertically.

        texture_bias (float): If positive, trim the edge of the texture by this
            amount to avoid artifacts.
    """
    if os.path.splitext(file_name)[1]:
        raise ValueError(
            'file_name to save_textured_mesh cannot contain extension')

    if file_name.find(' ') != -1:
        raise ValueError('file_name cannot contain space')

    obj_path = os.path.join(directory, file_name + '.obj')
    mtl_path = os.path.join(directory, file_name + '.mtl')
    texture_path = os.path.join(directory, file_name + '.png')

    with open(obj_path, 'w') as obj_file:
        obj_file.write('mtllib ./{}.mtl\n'.format(file_name))

        for pos in vertex_pos_px3:
            obj_file.write('v {} {} {}\n'.format(pos[0], pos[1], pos[2]))

        for uv in tex_coord_px2:
            uv = uv * 0.5 + 0.5  # normalize from [-1, 1] to [0, 1]
            uv = uv * (1.0 - texture_bias * 2.0) + texture_bias
            obj_file.write('vt {} {}\n'.format(
                uv[0],
                1.0 - uv[1] if flip_vertical else uv[1]
            ))

        obj_file.write('usemtl material_0\n')

        for i in range(face_fx3.shape[0]):
            face = face_fx3[i] + 1
            obj_file.write(
                'f {0}/{0} {1}/{1} {2}/{2}\n'.format(face[0], face[1], face[2]))

    with open(mtl_path, 'w') as mtl_file:
        mtl_file.write('''newmtl material_0
Ka 0.200000 0.200000 0.200000
Kd 1.000000 1.000000 1.000000
Ks 1.000000 1.000000 1.000000
map_Kd {}.png'''.format(file_name))

    cv2.imwrite(texture_path, cv2.cvtColor(
        (texture_hxwxc * 255).astype(np.float32), cv2.COLOR_RGB2BGR))

    return


def saveobjscale(meshfile, scale, maxratio, shift=None):

    mname, prefix = os.path.splitext(meshfile)
    mnamenew = '%s-%.2f%s' % (mname, maxratio, prefix)

    meshfp = open(meshfile, 'r')
    meshfp2 = open(mnamenew, 'w')
    for line in meshfp.readlines():
        data = line.strip().split(' ')
        data = [da for da in data if len(da) > 0]
        if len(data) != 4:
            meshfp2.write(line)
            continue
        else:
            if data[0] == 'v':
                p = [scale * float(d) for d in data[1:]]
                meshfp2.write('v %f %f %f\n' % (p[0], p[1], p[2]))
            else:
                meshfp2.write(line)
                continue

    meshfp.close()
    meshfp2.close()

    return


# 3
if __name__ == '__main__':

    meshjson = '1.obj'

    # f begin from 0!!!
    pointnp_px3, facenp_fx3 = loadobj(meshjson)
    assert np.max(facenp_fx3) == pointnp_px3.shape[0] - 1
    assert np.min(facenp_fx3) == 0

    pointnp_px3[:, 1] -= 0.05
    X = pointnp_px3[:, 0]
    Y = pointnp_px3[:, 1]
    Z = pointnp_px3[:, 2]
    h = 248 * (Y / Z) + 111.5
    w = -248 * (X / Z) + 111.5

    height = 224
    width = 224
    im = np.zeros(shape=(height, width), dtype=np.uint8)
    for cir in zip(w, h):
        cv2.circle(im, (int(cir[0]), int(cir[1])), 3, (255, 0, 0), -1)
    cv2.imshow('', im)
    cv2.waitKey()

    # edge, neighbour and pfmtx
    edgenp_ex2 = face2edge(facenp_fx3)

    face_edgeidx_fx3 = face2edge2(facenp_fx3, edgenp_ex2)

    pneimtx = face2pneimtx(facenp_fx3)
    pfmtx = face2pfmtx(facenp_fx3)

    # save
    savemesh(pointnp_px3, facenp_fx3, '1s.obj')
