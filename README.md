# Fabrics

Fabrics are nonlinear, autonomous, second order differential equations that are provably stable and exhibit path consistency properties. This library brings fabrics to the GPU enabling large-scale parallelization, differentiability, and more. The intended domain is robot control and we include a manually designed fabric policy for the Kuka-Allegro robot that enabled DextrAH training.

![](./docs/img/gifs/fabrics_collage.gif)

## Installation
1. [Install](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html) Isaac Sim, Isaac Lab following the local conda install route.

**Note**: After you clone the Isaac Lab repository and before installation, checkout the tag `v2.2.1` before installation (can also work with `v2.0.2` with minor code changes):
```bash
        cd <IsaacLab>
        git checkout v2.2.1
```
2. Install Fabrics within your new conda env
```bash
        curl -sSL https://install.python-poetry.org | python3 - --version 1.8.3
        git lfs clone git@github.com:NVlabs/FABRICS.git
        cd <FABRICS>
        poetry install
        or
        python -m pip install -e .
```
3. Patch `urdfpy` dependency `networkx`
```bash
       cd <fabrics_dir>
       chmod +x urdfpy_patch.sh
       ./urdfpy_patch.sh
```
## Examples
Once installed, you should be able to run the example script in the example directory, e.g.:

Kuka-Allegro (DextrAH) fabric:

    python <fabrics_dir>/examples/kuka_allegro_pose_fabric_example.py --batch_size=10 --render --cuda_graph

## Notes
One can update dependences in deps.txt file, remove pyproject.toml and poetry.lock files, and regenerate them
```bash
    cd <fabrics_dir>
    rm pyproject.toml poetry.lock
    poetry init --name "fabrics_sim" --no-interaction
    xargs poetry add < deps.txt
```
Add the following after ```authors``` in pyproject.toml
```bash
    packages = [ 
    { include = "fabrics_sim", from = "src" }
    ]
```
Install the project
```bash
    poetry install
```
