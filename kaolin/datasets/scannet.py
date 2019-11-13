from typing import Callable, Optional

import os
from collections import OrderedDict
import torch.utils.data as data
from torchvision import transforms


class ScanNet(data.Dataset):
    r"""ScanNet dataset http://www.scan-net.org/

    Args:
        root_dir (str): Path to the base directory of the dataset.
        scene_file (str): Path to file containing a list of scenes
            to be loaded.
        transform (callable, optional): A function/transform that takes in a PIL
            image and returns a transformed version of the image (default: None).
        label_transform (callable, optional): A function/transform that takes
            in the target and transforms it. (default: None).
        loader (callable, optional): A function to load an image given its path.
            By default, ``default_loader`` is used.
        color_mean (list): A list of length 3, containing the R, G, B channelwise
            mean.
        color_std (list): A list of length 3, containing the R, G, B channelwise
            standard deviation.
        load_depth (bool): Whether or not to load depth images (architectures
            that use depth information need depth to be loaded).
        seg_classes (string): The palette of classes that the network should
            learn.

    """

    def __init__(self, root_dir: str, scene_id: str,
                 mode: Optional[str] = 'inference',
                 transform: Optional[Callable] = None,
                 label_transform: Optional[Callable] = None,
                 loader: Optional[Callable] = None,
                 color_mean: Optional[list] = [0.,0.,0.],
                 color_std: Optional[list] = [1.,1.,1.],
                 load_depth: Optional[bool] = False,
                 seg_classes: Optional[str] = 'nyu40'):
        self.root_dir = root_dir
        self.scene_id = scene_id
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.length = 0
        self.color_mean = color_mean
        self.color_std = color_std
        self.load_depth = load_depth
        self.seg_classes = seg_classes
        # color_encoding has to be initialized AFTER seg_classes
        self.color_encoding = self.get_color_encoding()

        if self.loader is None:
            if self.load_depth is True:
                self.loader = self.scannet_loader_depth
            else:
                self.loader = self.scannet_loader

        # Get test data and labels filepaths
        self.data, self.depth, self.labels = get_filenames_scannet(
            self.root_dir, self.scene_id)
        self.length += len(self.data)        

    def __getitem__(self, index):
        """ Returns element at index in the dataset.

        Args:
            index (``int``): index of the item in the dataset

        Returns:
            A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image

        """

        if self.load_depth is True:

            data_path, depth_path, label_path = self.data[index], self.depth[index], self.labels[index]
            rgbd, label = self.loader(data_path, depth_path, label_path, self.color_mean, self.color_std, \
                self.seg_classes)
            return rgbd, label, data_path, depth_path, label_path

        else:

            data_path, label_path = self.data[index], self.labels[index]
            img, label = self.loader(data_path, label_path, self.color_mean, self.color_std, self.seg_classes)

            return img, label, data_path, label_path

    def __len__(self):
        """ Returns the length of the dataset. """
        return self.length

    def get_color_encoding(self):
        if self.seg_classes.lower() == 'nyu40':
            """Color palette for nyu40 labels """
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('blinds', (178, 76, 76)),
                ('desk', (247, 182, 210)),
                ('shelves', (66, 188, 102)),
                ('curtain', (219, 219, 141)),
                ('dresser', (140, 57, 197)),
                ('pillow', (202, 185, 52)),
                ('mirror', (51, 176, 203)),
                ('floormat', (200, 54, 131)),
                ('clothes', (92, 193, 61)),
                ('ceiling', (78, 71, 183)),
                ('books', (172, 114, 82)),
                ('refrigerator', (255, 127, 14)),
                ('television', (91, 163, 138)),
                ('paper', (153, 98, 156)),
                ('towel', (140, 153, 101)),
                ('showercurtain', (158, 218, 229)),
                ('box', (100, 125, 154)),
                ('whiteboard', (178, 127, 135)),
                ('person', (120, 185, 128)),
                ('nightstand', (146, 111, 194)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('lamp', (96, 207, 209)),
                ('bathtub', (227, 119, 194)),
                ('bag', (213, 92, 176)),
                ('otherstructure', (94, 106, 211)),
                ('otherfurniture', (82, 84, 163)),
                ('otherprop', (100, 85, 144)),
            ])
        elif self.seg_classes.lower() == 'scannet20':
            return OrderedDict([
                ('unlabeled', (0, 0, 0)),
                ('wall', (174, 199, 232)),
                ('floor', (152, 223, 138)),
                ('cabinet', (31, 119, 180)),
                ('bed', (255, 187, 120)),
                ('chair', (188, 189, 34)),
                ('sofa', (140, 86, 75)),
                ('table', (255, 152, 150)),
                ('door', (214, 39, 40)),
                ('window', (197, 176, 213)),
                ('bookshelf', (148, 103, 189)),
                ('picture', (196, 156, 148)),
                ('counter', (23, 190, 207)),
                ('desk', (247, 182, 210)),
                ('curtain', (219, 219, 141)),
                ('refrigerator', (255, 127, 14)),
                ('showercurtain', (158, 218, 229)),
                ('toilet', (44, 160, 44)),
                ('sink', (112, 128, 144)),
                ('bathtub', (227, 119, 194)),
                ('otherfurniture', (82, 84, 163)),
            ])

    def get_filenames_scannet(base_dir: str, scene_id: str):
        """Helper function that returns a list of scannet images and the
        corresponding segmentation labels, given a base directory name
        and a scene id.

        Args:
        base_dir (str): Path to the base directory containing ScanNet
            data, in the directory structure specified in
            https://github.com/angeladai/3DMV/tree/master/prepare_data
        scene_id (str): ScanNet scene id

        """

        if not os.path.isdir(base_dir):
            raise RuntimeError('\'{0}\' is not a directory.'.format(base_dir))

        color_images = []
        depth_images = []
        labels = []

        # Explore the directory tree to get a list of all files
        for path, _, files in os.walk(os.path.join(
                                      base_dir, scene_id, 'color')):
            files = natsorted(files)
            for file in files:
                filename, _ = os.path.splitext(file)
                depthfile = os.path.join(base_dir, scene_id, 'depth',
                    filename + '.png')
                labelfile = os.path.join(base_dir, scene_id, 'label',
                    filename + '.png')
                # Add this file to the list of train samples, only if its
                # corresponding depth and label files exist.
                if os.path.exists(depthfile) and os.path.exists(labelfile):
                    color_images.append(os.path.join(base_dir, scene_id,
                        'color', filename + '.jpg'))
                    depth_images.append(depthfile)
                    labels.append(labelfile)

        # Assert that we have the same number of color, depth images as labels
        assert (len(color_images) == len(depth_images) == len(labels))

        return color_images, depth_images, labels

    def get_files(self, folder: str, name_filter: Optional[str] = None,
                  extension_filter: Optional[str] = None):
        """Helper function that returns the list of files in a specified folder
        with a specified extension.

        Args:
        folder (str): The path to a folder.
        name_filter (str, optional): The returned files must contain
            this substring in their filename (default: None, files are
            not filtered).
        extension_filter (str, optional): The desired file extension
            (default: None; files are not filtered).

        """
        if not os.path.isdir(folder):
            raise RuntimeError("\"{0}\" is not a folder.".format(folder))

        # Filename filter: if not specified don't filter (condition always
        # true); otherwise, use a lambda expression to filter out files that
        # do not contain "name_filter"
        if name_filter is None:
            # This looks hackish...there is probably a better way
            name_cond = lambda filename: True
        else:
            name_cond = lambda filename: name_filter in filename

        # Extension filter: if not specified don't filter (condition always
        # true); otherwise, use a lambda expression to filter out files whose
        # extension is not "extension_filter"
        if extension_filter is None:
            # This looks hackish...there is probably a better way
            ext_cond = lambda filename: True
        else:
            ext_cond = lambda filename: filename.endswith(extension_filter)

        filtered_files = []

        # Explore the directory tree to get files that contain "name_filter"
        # and with extension "extension_filter"
        for path, _, files in os.walk(folder):
            files.sort()
            for file in files:
                if name_cond(file) and ext_cond(file):
                    full_path = os.path.join(path, file)
                    filtered_files.append(full_path)

        return filtered_files

    def scannet_loader(self, data_path: str, label_path: str,
                       color_mean: Optional[list] = [0.,0.,0.],
                       color_std: Optional[list] = [1.,1.,1.],
                       seg_classes: str = 'nyu40'):
        """Loads a sample and label image given their path as PIL images
        (nyu40 classes).

        Args:
        data_path (str): The filepath to the image.
        label_path (str): The filepath to the ground-truth image.
        color_mean (str): R, G, B channel-wise mean
        color_std (str): R, G, B channel-wise stddev
        seg_classes (str): Palette of classes to load labels for
            ('nyu40' or 'scannet20')

        Returns the image and the label as PIL images.

        """

        # Load image.
        data = np.array(imageio.imread(data_path))
        # Reshape data from H x W x C to C x H x W.
        data = np.moveaxis(data, 2, 0)
        # Define normalizing transform.
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # Convert image to float and map range from [0, 255] to [0.0, 1.0].
        # Then normalize.
        data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

        # Load label.
        if seg_classes.lower() == 'nyu40':
            label = np.array(imageio.imread(label_path)).astype(np.uint8)
        elif seg_classes.lower() == 'scannet20':
            label = np.array(imageio.imread(label_path)).astype(np.uint8)
            # Remap classes from 'nyu40' to 'scannet20'
            label = self.nyu40_to_scannet20(label)

        return data, label


    def scannet_loader_depth(self, data_path: str, depth_path: str,
                             label_path: str,
                             color_mean: Optional[list] = [0.,0.,0.],
                             color_std: Optional[list] = [1.,1.,1.],
                             seg_classes: Optional[str] = 'nyu40'):
        """Loads a sample and label image given their path as PIL images
        (nyu40 classes).

        Args:
        data_path (str): The filepath to the image.
        depth_path (str): The filepath to the depth png.
        label_path (str): The filepath to the ground-truth image.
        color_mean (list): R, G, B channel-wise mean.
        color_std (list): R, G, B channel-wise stddev.
        seg_classes (str): Palette of classes to load labels for
            ('nyu40' or 'scannet20').

        Returns:
            (PIL.Image): the image
            (PIL.Image): the label as PIL images.

        """

        # Load image
        rgb = np.array(imageio.imread(data_path))
        # Reshape rgb from H x W x C to C x H x W
        rgb = np.moveaxis(rgb, 2, 0)
        # Define normalizing transform
        normalize = transforms.Normalize(mean=color_mean, std=color_std)
        # Convert image to float and map range from [0, 255] to [0.0, 1.0].
        # Then normalize.
        rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))

        # Load depth
        depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(
            np.float32) / 1000.0)
        depth = torch.unsqueeze(depth, 0)

        # Concatenate rgb and depth
        data = torch.cat((rgb, depth), 0)

        # Load label
        if seg_classes.lower() == 'nyu40':
            label = np.array(imageio.imread(label_path)).astype(np.uint8)
        elif seg_classes.lower() == 'scannet20':
            label = np.array(imageio.imread(label_path)).astype(np.uint8)
            # Remap classes from 'nyu40' to 'scannet20'
            label = self.nyu40_to_scannet20(label)

        return data, label


    def nyu40_to_scannet20(self, label: str):
        """Remap a label image from the 'nyu40' class palette to the
        'scannet20' class palette """

        # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29.
        # 30. 31. 32, 35. 37. 38, 40
        # Because, these classes from 'nyu40' are absent from 'scannet20'.
        # Our label files are in 'nyu40' format, hence this 'hack'.
        # To see detailed class lists visit:
        # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt
        # (for 'nyu40' labels), and
        # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt
        # (for 'scannet20' labels).
        # The remaining labels are then to be mapped onto a contiguous
        # ordering in the range [0,20].

        # The remapping array comprises tuples (src, tar), where 'src'
        # is the 'nyu40' label, and 'tar' is the corresponding target
        # 'scannet20' label.
        remapping = [(0, 0), (13, 0), (15, 0), (17, 0), (18, 0), (19, 0),
                     (20, 0), (21, 0), (22, 0), (23, 0), (25, 0), (26, 0),
                     (27, 0), (29, 0), (30, 0), (31, 0), (32, 0), (35, 0),
                     (37, 0), (38, 0), (40, 0), (14, 13), (16, 14), (24, 15),
                     (28, 16), (33, 17), (34, 18), (36, 19), (39, 20)]
        for src, tar in remapping:
            label[np.where(label==src)] = tar
        return label

    def create_label_image(output, color_palette):
        """Create a label image, given a network output (each pixel contains
        # class index) and a color palette.

        Args:
        output (np.array, dtype = np.uint8): Output image. Height x Width.
            Each pixel contains an integer, corresponding to the class label
            for that pixel.
        color_palette (OrderedDict): Contains (R, G, B) colors (uint8)
            for each class.

        """
        
        label_image = np.zeros((output.shape[0], output.shape[1], 3),
                               dtype=np.uint8)
        for idx, color in enumerate(color_palette):
            label_image[output==idx] = color
        return label_image
