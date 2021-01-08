
# Kaolin Dash3D (beta)

WebGL dashboard that allows viewing checkpoints written
with kaolin's `Timelapse` from any remote machine in a web browser. 
Kaolin Dash3D is a **lightweight alternative** to the Training Visualizer
within the 
[Kaolin Omniverse App](https://docs.omniverse.nvidia.com/app_kaolin/app_kaolin/user_manual.html),
which is a desktop application with wider visualization support
and superior rendering quality. 

Currently Dash3D can only display geometric data of the following types:

* 3d meshes (no texture or vertex color support)
* point clouds (no color)

Voxelgrid, as well as texture and color support is planned.


## Using

Simply run `kaolin-dash3d` on the machine that has your training checkpoints:

```
$ kaolin-dash3d --logdir=$MY_EXPERIMENTS/test0/checkpts --port=8080
```

Then, view the 3D checkpoints in a browser window locally by going to
[localhost:8080](http://localhost:8080/), or connect to the server remotely.
(Tip: use ssh port forwarding on your client machine.)

## Bundled Third-party code

We would like to acknowledge the following third-party code bundled with
this application:
* [jQuery](https://jquery.com/download/), released under [MIT License](https://jquery.org/license/) 
* [ThreeJS](https://github.com/mrdoob/three.js/), released under [MIT License](https://github.com/mrdoob/three.js/blob/dev/LICENSE)
* [Mustard UI](https://kylelogue.github.io/mustard-ui/index.html), released under [MIT License](https://github.com/kylelogue/mustard-ui/blob/master/LICENSE)

## Developing

#### Dependencies

Third party dependencies for the javascript frontend as well as for 
testing are managed by `npm`. To fetch dependencies, first 
install `nodejs`, which is available through `conda`, and run `npm install`
from the root of kaolin (this will install both production and development
dependencies). 

To add new dependencies, run this from the root of kaolin:
```
npm install <PKG> --save-dev
```
This will also update the `package.json` file in the root of kaolin.
If the new dependencies must be served to the client, omit 
`--save-dev` from the command, and add the appropriate source to
the `Makefile` in this directory. 

#### Compiling client side code

Compiled source and thirdparty javascript is bundled with the app in the 
`static` subdirectory, while source javascript and CSS code is
located under `src/`. After edits, it
must be compiled into an optimized version to be served:
```
$ cd kaolin/visualize/dash3d
$ make clean
$ make
```
Note that this also compiles thirdparty dependencies which are
assumed to reside in `node-modules` in the root of `kaolin`, a 
directory created automatically by `npm`. This is not ideal,
but for now we avoid setting up tools like WebPack, 
given our small set of dependencies.

#### Integration testing

Integration tests are located in `kaolin/tests/integration`. Currently
the set of implemented tests is limited and excluded from continuous 
integration due to their complexity. Please refer to the README
in the test directory.
