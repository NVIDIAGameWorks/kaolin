"""
Utils to generate data to test toy 3d super-resolution
"""

import argparse
import numpy as np
import os
from tqdm import tqdm, trange

from genutils import rotx, roty, rotz, points2voxels


parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='def', help='Unique experiment identifier.')
parser.add_argument('-numsamples', type=int, default=5, help='Number of samples to generate.')
parser.add_argument('-numpts', type=int, default=100, help='Number of points in each sample.')
parser.add_argument('-debug', action='store_true', help='Debug mode (for visualization).')
parser.add_argument('-savedir', type=str, default='data', \
	help='Directory to store generated samples.')
parser.add_argument('-randomseed', type=int, default=12, help='Seed random number generator.')
parser.add_argument('-rad-min', type=float, default=0.1, \
	help='Minimum radius of the generated ellipsoid.')
parser.add_argument('-rad-max', type=float, default=1., \
	help='Maximum radius of the generated ellipsoid.')
parser.add_argument('-res-in', type=int, default=16, help='Resolution of input voxel grid.')
parser.add_argument('-res-out', type=int, default=32, help='Resolution of output voxel grid.')
parser.add_argument('-voxsize', type=float, default=0.2, help='Size of each voxel grid cell.')
parser.add_argument('-rot-min', type=float, default=-1.5, help='Rotation range minimum.')
parser.add_argument('-rot-max', type=float, default=1.5, help='Rotation range maximum.')
parser.add_argument('-trans-min', type=float, default=-0.5, help='Translation range minimum.')
parser.add_argument('-trans-max', type=float, default=0.5, help='Translation range maximum.')


def sample_on_ellipsoid(numpts, radius=(1.,1.,1.), rot=(0.,0.,0.), trans=(0.,0.,0.)):
	"""Sample points uniformly randomly on the surface of an ellipsoid.

	Args:
		- numpts (int): Number of points to sample
		- radius (tuple, float): radius of each axis (X, Y, Z) of the ellipsoid
		- rot (np.array): rotation parameters (ZYX Euler angles)
		- trans (np.array): translation vector
	"""

	assert isinstance(numpts, int), 'Integer required'

	phi = np.random.uniform(0, np.pi, numpts)
	theta = np.random.uniform(0, 2*np.pi, numpts)
	x = np.expand_dims(radius[0] * np.cos(theta) * np.sin(phi), axis=1)
	y = np.expand_dims(radius[1] * np.sin(theta) * np.sin(phi), axis=1)
	z = np.expand_dims(radius[2] * np.cos(phi), axis=1)
	pts = np.concatenate([x, y, z], axis=1)

	# NOTE: Scaling is not really required (redundant as we already perturb the radius)
	# Rotate and translate the ellipsoid
	rotmat = rotz(rot[0]).dot(roty(rot[1]).dot(rotx(rot[0])))
	pts = rotmat.dot(pts.T).T + np.reshape(np.asarray(trans), (1,3))

	# Retain only the number of points that are required
	idx = np.round(np.linspace(1, pts.shape[0]-1, numpts)).astype(int)
	pts = pts[idx,:]

	return pts


def generate_ellipsoid_voxels(radius, rot, trans, res_in=16, res_out=32, voxsize=1.):
	"""Generate a voxel grid containing an ellipsoid. One low-res grid, corresponding to 
	the input, and it's respective hi-res version.

	Params:
		- radius (tuple, float): radius of each axis (X, Y, Z) of the ellipsoid
		- rot (np.array): rotation parameters (ZYX Euler angles)
		- trans (np.array): translation vector
		- res_in (int): resolution of the input voxel grid
		- res_out (int): resolution of the output voxel grid
		- voxsize (float): size of each voxel grid cell
	"""

	pts = sample_on_ellipsoid(5000, radius=radius, rot=rot, trans=trans)
	in_voxels = points2voxels(pts, res_in, voxsize)
	out_voxels = points2voxels(pts, res_out, voxsize)
	return pts, in_voxels, out_voxels


if __name__ == '__main__':

	# Parse commandline args
	args = parser.parse_args()

	# Seed random number generator (for repeatability)
	np.random.seed(args.randomseed)

	# Debug mode (visualize)
	if args.debug:
		
		# Generate points on ellipsoid
		# pts = sample_on_ellipsoid(args.numpts, radius=(1.,2.,1.), rot=(0.86,0.,0.), \
		# 	trans=(1.0,0.5,0.))
		pts, in_voxels, out_voxels = generate_ellipsoid_voxels(radius=(1.,2.,1.), \
			rot=(0.86,0.,0.), trans=(0.2,0.3,0), voxsize=0.2)
		
		import matplotlib
		from matplotlib import pyplot as plt
		from mpl_toolkits.mplot3d import axes3d
		fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
		ax.scatter(pts[:,0], pts[:,1], pts[:,2], c='r')
		plt.show()

		import sys
		sys.exit(0)

	# Create dir to save samples, if it does not already exist
	args.savedir = os.path.join(args.savedir, args.expid)
	savedir_pts = os.path.join(args.savedir, 'pts_' + str(args.numpts).zfill(4))
	savedir_vox_in = os.path.join(args.savedir, 'vox_' + str(args.res_in).zfill(4))
	savedir_vox_out = os.path.join(args.savedir, 'vox_' + str(args.res_out).zfill(4))
	if not os.path.isdir(args.savedir):
		os.makedirs(args.savedir)
		print('Created dir:', args.savedir)
		os.makedirs(savedir_pts)
		os.makedirs(savedir_vox_in)
		os.makedirs(savedir_vox_out)
		print('Created subdirs.')

	# Generate samples
	for i in trange(args.numsamples):
		# Sample a radius
		rad = (np.random.uniform(args.rad_min, args.rad_max), \
			np.random.uniform(args.rad_min, args.rad_max), \
			np.random.uniform(args.rad_min, args.rad_max))
		rot = (np.random.uniform(args.rot_min, args.rot_max), \
			np.random.uniform(args.rot_min, args.rot_max), \
			np.random.uniform(args.rot_min, args.rot_max))
		trans = (np.random.uniform(args.trans_min, args.trans_max), \
			np.random.uniform(args.trans_min, args.trans_max), \
			np.random.uniform(args.trans_min, args.trans_max))
		
		# Generate ellipsoid data sample
		pts, in_voxels, out_voxels = generate_ellipsoid_voxels(radius=rad, rot=rot, \
			trans=trans, res_in=args.res_in, res_out=args.res_out, voxsize=args.voxsize)
		# Save sample
		curfile = str(i).zfill(5)
		np.save(os.path.join(savedir_pts, curfile), pts)
		np.save(os.path.join(savedir_vox_in, curfile), in_voxels)
		np.save(os.path.join(savedir_vox_out, curfile), out_voxels)
