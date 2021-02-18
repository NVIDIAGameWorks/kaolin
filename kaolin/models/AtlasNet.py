import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import trimesh
from torchvision.models import resnet18
from torch.autograd import Variable
import numpy as np
import kaolin as kal

class AtlasNet(nn.Module):

    def __init__(self, opt):
        """
        Core AtlasNet module : decoder to meshes and pointclouds.
        This network takes an embedding in the form of a latent vector and returns a pointcloud or a mesh
        Author : Thibault Groueix 01.11.2019
        :param opt:
        """
        super(AtlasNet, self).__init__()
        self.opt = opt
        self.device = opt.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = opt.number_points // opt.nb_primitives
        self.nb_pts_in_primitive_eval = opt.number_points_eval // opt.nb_primitives

        if opt.remove_all_batchNorms:
            torch.nn.BatchNorm1d = Identity
            print("Replacing all batchnorms by identities.")

        # Initialize templates
        self.template = [get_template(opt.template_type, device=opt.device) for i in range(0, opt.nb_primitives)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList([Mapping2Dto3D(opt) for i in range(0, opt.nb_primitives)])

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an opt.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        # Sample points in the patches
        # input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive,
        #                                                     device=latent_vector.device)
        #                 for i in range(self.opt.nb_primitives)]
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                latent_vector.device) for i in range(self.opt.nb_primitives)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).unsqueeze(1) for i in
                                   range(0, self.opt.nb_primitives)], dim=1)

        # Return the deformed pointcloud
        return output_points.contiguous().transpose(2,3).contiguous().view(latent_vector.size(0), -1, 3)  # batch, nb_prim, num_point, 3

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"
        import trimesh
        input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval, latent_vector.device)
                        for i in range(self.opt.nb_primitives)]
        input_points = [input_points[i] for i in range(self.opt.nb_primitives)]

        # Deform each patch
        output_points = [self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze() for i in
                         range(0, self.opt.nb_primitives)]

        output_meshes = [trimesh.Trimesh(vertices=output_points[i].transpose(1, 0).contiguous().cpu().numpy(),
                                          faces=self.template[i].mesh.faces)
                         for i in range(self.opt.nb_primitives)]

        return output_meshes


class Mapping2Dto3D(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, opt):
        self.opt = opt
        self.bottleneck_size = opt.bottleneck_size
        self.input_size = opt.dim_template
        self.dim_output = 3
        self.hidden_neurons = opt.hidden_neurons
        self.num_layers = opt.num_layers
        super(Mapping2Dto3D, self).__init__()
#         print(
#             f"New MLP decoder : hidden size {opt.hidden_neurons}, num_layers {opt.num_layers}, activation {opt.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(opt.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.opt.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)







# Templates
"""
        Author : Thibault Groueix 01.11.2019
"""


def get_template(template_type, device=0):
    getter = {
        "SQUARE": SquareTemplate,
        "SPHERE": SphereTemplate,
    }
    template = getter.get(template_type, "Invalid template")
    return template(device=device)


class Template(object):
    def get_random_points(self):
        print("Please implement get_random_points ")

    def get_regular_points(self):
        print("Please implement get_regular_points ")


class SphereTemplate(Template):
    def __init__(self, device=0, grain=6):
        self.device = device
        self.dim = 3
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 3, x ... x]
        """
        assert shape[1] == 3, "shape should have 3 in dim 1"
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.normal_(0, 1)
        rand_grid = rand_grid / torch.sqrt(torch.sum(rand_grid ** 2, dim=1, keepdim=True))
        return Variable(rand_grid)

    def get_regular_points(self, npoints=None, device="gpu0"):
        """
        Get regular points on a Sphere
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.mesh = trimesh.generate_icosphere(1, [0, 0, 0], 4)  # 2562 vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)
            self.npoints = npoints

        return Variable(self.vertex.to(device))


class SquareTemplate(Template):
    def __init__(self, device=0):
        self.device = device
        self.dim = 2
        self.npoints = 0

    def get_random_points(self, shape, device="gpu0"):
        """
        Get random points on a Sphere
        Return Tensor of Size [x, 2, x ... x]
        """
        rand_grid = torch.cuda.FloatTensor(shape).to(device).float()
        rand_grid.data.uniform_(0, 1)
        return Variable(rand_grid)

    def get_regular_points(self, npoints=2500, device="gpu0"):
        """
        Get regular points on a Square
        Return Tensor of Size [x, 3]
        """
        if not self.npoints == npoints:
            self.npoints = npoints
            vertices, faces = self.generate_square(np.sqrt(npoints))
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)  # 10k vertices
            self.vertex = torch.from_numpy(self.mesh.vertices).to(device).float()
            self.num_vertex = self.vertex.size(0)
            self.vertex = self.vertex.transpose(0,1).contiguous().unsqueeze(0)

        return Variable(self.vertex[:, :2].contiguous().to(device))

    @staticmethod
    def generate_square(grain):
        """
        Generate a square mesh from a regular grid.
        :param grain:
        :return:
        """
        grain = int(grain)
        grain = grain - 1  # to return grain*grain points
        # generate regular grid
        faces = []
        vertices = []
        for i in range(0, int(grain + 1)):
            for j in range(0, int(grain + 1)):
                vertices.append([i / grain, j / grain, 0])

        for i in range(1, int(grain + 1)):
            for j in range(0, (int(grain + 1) - 1)):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i + 1,
                              j + (grain + 1) * (i - 1)])
        for i in range(0, (int((grain + 1)) - 1)):
            for j in range(1, int((grain + 1))):
                faces.append([j + (grain + 1) * i,
                              j + (grain + 1) * i - 1,
                              j + (grain + 1) * (i + 1)])

        return np.array(vertices), np.array(faces)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")



## From Kaolin.examples.ImageRecon.OccNet.architecture
class Resnet18(nn.Module):
    r''' ResNet-18 encoder network for image input.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = resnet18(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x