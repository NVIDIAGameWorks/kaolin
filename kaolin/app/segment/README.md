# Sample App: Gaussian Splat Segmentation

This app sets up a **client–server interactive application** for **segmenting Gaussian
splats**, combining server-side splat rendering with interactive 2D selection and
**SAM2**-based point-prompt segmentation.

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
pip install -e ".[app.segment]" --no-build-isolation
```

> ℹ️ **Note:** We probably won't ship apps like this — this is a placeholder solution to
> easily share development.

---

## 3. Running the App

To see the available arguments, run:

```bash
python -m kaolin.app.segment.main --help
```

Minimal run example:

```bash
python -m kaolin.app.segment.main --input_scene=/path/to/scene.ply
```

Then open: [localhost:8001](http://localhost:8001)

---

## 4. API Walk-Through

Read the comments in [`kaolin/app/segment/main.py`](./main.py) for a walk-through of the API.

---

## 5. Usage

See [`USAGE.md`](./USAGE.md) for a quick guide to the available modes.

<!-- @import "USAGE.md" -->
