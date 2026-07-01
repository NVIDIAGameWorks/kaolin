# Tutorial App Installation

Start with an activated Conda environment using Python 3.11+ and an appropriate
PyTorch/CUDA installation. 

```bash
conda create --name kaolin-app python=3.11 -y
conda activate kaolin-app
python -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

From the Kaolin repository root, run:

```bash
git submodule update --init --recursive
conda install -c conda-forge nodejs=26 -y
python -m pip install \
  -r tools/build_requirements.txt \
  -r tools/viz_requirements.txt \
  -r tools/requirements.txt
python setup.py develop

# On memory-limited machines, avoid compilation running out of memory with:
# MAX_JOBS=1 python setup.py develop
```

Install and run one app at a time using the commands below.

*Note: Applications may take a few minutes to build the first time they are loaded.*

## Applications

### segment

```bash
python -m pip install -r kaolin/app/segment/requirements.txt

python -m kaolin.app.segment.main \
  --input_scene="$PWD/sample_data/scanned_toys/sunflower_baby.ply"
```

Open <http://localhost:8001>.

### mesh_edit

This app requires a separate Axolotl3D installation. Follow
[`mesh_edit/README.md`](mesh_edit/README.md), then run:

```bash
python -m kaolin.app.mesh_edit.main
```

Open <http://localhost:8000>.
