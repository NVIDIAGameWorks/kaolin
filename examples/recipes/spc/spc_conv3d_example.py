# ==============================================================================================================
# The following code demonstrates the usage of kaolin's "Structured Point Cloud (SPC)" 3d convolution
# functionality. Note that this sample does NOT demonstrate how to use Kaolin's Pytorch 3d convolution layers. 
# Rather, 3d convolutions are used to 'filter' color data useful for level-of-detail management during 
# rendering. This can be thought of as the 3d analog of generating a 2d mipmap. 
# 
# Note this is a low level interface: practitioners are encouraged to visit the references below.
# ==============================================================================================================
# See also:
#
#  - Code: kaolin.ops.spc.SPC
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.rep.html?highlight=SPC#kaolin.rep.Spc
#
#  - Tutorial: Understanding Structured Point Clouds (SPCs)
#    https://github.com/NVIDIAGameWorks/kaolin/blob/master/examples/tutorial/understanding_spcs_tutorial.ipynb
#
#  - Documentation: Structured Point Clouds
#    https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=spc#kaolin-ops-spc
# ==============================================================================================================

import torch
import kaolin

# The following function applies a series of SPC convolutions to encode the entire hierarchy into a single tensor.
# Each step applies a convolution on the "highest" level of the SPC with some averaging kernel.
# Therefore, each step locally averages the "colored point hierarchy", where each "colored point"
# corresponds to a point in the SPC point hierarchy.
# For a description of inputs 'octree', 'point_hierachy', 'level', 'pyramids', and 'exsum', as well a
# detailed description of the mathematics of SPC convolutions, see:
# https://kaolin.readthedocs.io/en/latest/modules/kaolin.ops.spc.html?highlight=SPC#kaolin.ops.spc.Conv3d
# The input 'color' is Pytorch tensor containing color features corresponding to some 'level' of the hierarchy.
def encode(colors, octree, point_hierachy, pyramids, exsum, level):

    # SPC convolutions are characterized by a set of 'kernel vectors' and corresponding 'weights'.

    # kernel_vectors is the "kernel support" -
    # a listing of 3D coordinates where the weights of the convolution are non-null,
    # in this case a it's a simple dense 2x2x2 grid.
    kernel_vectors = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1],
                                   [1,0,0],[1,0,1],[1,1,0],[1,1,1]], 
                                   dtype=torch.short, device='cuda')

    # The weights specify how the input colors 'under' the kernel are mapped to an output color, 
    # in this case a simple average.
    weights = torch.diag(torch.tensor([0.125, 0.125, 0.125, 0.125],
                                      dtype=torch.float32, device='cuda'))  # Tensor of (4, 4)
    weights = weights.repeat(8,1,1).contiguous()  # Tensor of (8, 4, 4)

    # Storage for the output color hierarchy is allocated. This includes points at the bottom of the hierarchy, 
    # as well as intermediate SPC levels (which may store different features)
    color_hierarchy = torch.empty((pyramids[0,1,level+1],4), dtype=torch.float32, device='cuda')
    # Copy the input colors into the highest level of color_hierarchy. pyramids is used here to select all leaf 
    # points at the bottom of the hierarchy and set them to some pre-sampled random color. Points at intermediate 
    # levels are left empty.
    color_hierarchy[pyramids[0,1,level]:pyramids[0,1,level+1]] = colors[:]

    # Performs the 3d convolutions in a bottom up fashion to 'filter' colors from the previous level
    for l in range(level,0,-1):

        # Apply the 3d convolution. Note that jump=1 means the inputs and outputs differ by 1 level
        # This is analogous to to a stride=2 in grid based convolutions
        colors, ll = kaolin.ops.spc.conv3d(octree, 
                                           point_hierachy, 
                                           l, 
                                           pyramids, 
                                           exsum, 
                                           colors, 
                                           weights, 
                                           kernel_vectors, 
                                           jump=1)
        # Copy the output colors into the color hierarchy
        color_hierarchy[pyramids[0,1,ll]:pyramids[0,1,l]] = colors[:]
        print(f"At level {l}, output feature shape is:\n{colors.shape}")

    # Normalize the colors. 
    color_hierarchy /= color_hierarchy[:,3:]
    # Normalization is needed here due to the sparse nature of SPCs. When a point under a kernel is not 
    # present in the point hierarchy, the corresponding data is treated as zeros. Normalization is equivalent 
    # to having the filter weights sum to one. This may not always be desirable, e.g. alpha blending.

    return color_hierarchy


# Highest level of SPC
level = 3

# Construct a fully occupied Structured Point Cloud with N levels of detail
# See https://kaolin.readthedocs.io/en/latest/modules/kaolin.rep.html?highlight=SPC#kaolin.rep.Spc
spc = kaolin.rep.Spc.make_dense(level, device='cuda')

# In kaolin, operations are batched by default, the spc object above contains a single item batch, hence [0]
num_points_last_lod = spc.num_points(level)[0]

# Create tensor of random colors for all points in the highest level of detail
colors = torch.rand((num_points_last_lod, 4), dtype=torch.float32, device='cuda')
# Set 4th color channel to one for subsequent color normalization
colors[:,3] = 1

print(f'Input SPC features: {colors.shape}')

# Encode color hierarchy by invoking a series of convolutions, until we end up with a single tensor.
color_hierarchy = encode(colors=colors,
                         octree=spc.octrees,
                         point_hierachy=spc.point_hierarchies,
                         pyramids=spc.pyramids,
                         exsum=spc.exsum,
                         level=level)

# Print root node color
print(f'Final encoded value (average of averages):')
print(color_hierarchy[0])
# This will be the average of averages, over the entire spc hierarchy. Since the initial random colors
# came from a uniform distribution, this should approach [0.5, 0.5, 0.5, 1.0] as 'level' increases
