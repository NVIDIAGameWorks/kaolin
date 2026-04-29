# Sample App: Toy Gaussian Splat Inpainter

This app sets up a **client–server interactive application** with **server-side
splat rendering**, **server-side 2D inpainting** via a diffusion pipeline, and
a per-Gaussian color optimization that **bakes 2D edits back into the 3D
Gaussian Splat scene**.

It also demonstrates key features of the new Kaolin UI/web toolkit in
[`kaolin/visualize`](../../visualize).

> 🚧 **Warning:** the `kaolin/visualize` is actively changing; this branch will be rebased regularly.

---

## 1. Installation (Kaolin Library)

1. Clone [internal gitlab Kaolin](https://gitlab-master.nvidia.com/Toronto_DL_Lab/kaolin) repository **recursively**:

   ```bash
   git clone --recursive <copy location from gitlab>
   ```

2. Check out **this branch**.

3. Follow the [Kaolin Documentation](https://kaolin.readthedocs.io/en/latest/notes/installation.html#installation-from-source) to create a new environment and build from source:

   - Use **Python >= 3.11**
   - Use **torch ~= 2.10**
   - Install **nodejs >= 25** into your environment before building Kaolin (e.g. `conda install -c conda-forge nodejs`)

> ⚠️ **Important!** If any dependencies are missing, let **Masha** know right away.

---

## 2. Installing the Sample App

To install this app's dependencies, run:

```bash
pip install -e ".[app.splat_inpaint]" --no-build-isolation
```

> ℹ️ **Note:** We probably won't ship apps like this — this is a placeholder solution to
> easily share development.

---

## 3. Running the App

To see the available arguments, run:

```bash
python -m kaolin.app.splat_inpaint.main --help
```

Minimal run example:

```bash
python -m kaolin.app.splat_inpaint.main --input_scene=/path/to/scene.ply
```

Then open: [localhost:8000](http://localhost:8000)

The viewer offers three modes (icons in the viewer's top-left, also bound to
keys `1`, `2`, `3`):

- **View** — interactive camera control over the server-rendered scene.
- **Mask** — paint a mask over the rendered image to mark the inpaint region.
- **Edit** — touch up the diffusion-inpainted result before baking it back into 3D.

Sidebar controls:

- **Inpaint** — sends the current view + mask + prompt to the server's
  diffusion pipeline; the inpainted RGBA is drawn back into the inpaint layer.
- **Clear** — clears the mask and inpaint layers.
- **Optimize in Mask** — runs the per-Gaussian color optimization that bakes
  the inpainted 2D pixels into the 3D splat scene.

---

## 4. API Walk-Through

Read the comments in [`kaolin/app/splat_inpaint/main.py`](./main.py) for a walk-through of the API.
See also the sibling example [`kaolin/app/act_splat/main.py`](../act_splat/main.py),
which is the canonical reference for the current `kaolin/visualize` API.

---

## 5. Sample PLYs (if needed)

To download samples, simply run the following (you can `export KAOLIN_SCANNED_TOYS_PATH=custom_path`
to control the download location):

```python
from kaolin.utils.bundled_data import (
    SCANNED_TOYS_PATH,
    SCANNED_TOYS_NAMES,
    download_scanned_toys_dataset,
)

download_scanned_toys_dataset()
```
