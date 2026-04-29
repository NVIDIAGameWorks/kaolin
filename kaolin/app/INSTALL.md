# Web App Installation

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

Install and run one app at a time, for example:

```bash
pip install -e ".[app.segment]" --no-build-isolation
```

See individual application documentation for further details.

*Note: Applications may take a few minutes to build the first time they are loaded.*

## Sample Applications

* [Gaussian Splat Segmentation](segment/README.md)
* [Toy Gaussian Splat Inpainter](splat_inpaint/README.md)



