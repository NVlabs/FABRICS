# Fabrics

Fabrics are nonlinear, autonomous, second order differential equations that are provably stable and exhibit path consistency properties. This library brings fabrics to the GPU enabling large-scale parallelization, differentiability, and more. The intended domain is robot control and, therefore, several manually derived fabric policies for some existing robots are included that enable reaching behavior while avoiding collision and respecting the various constraints of the robot. The construction of trainable neural fabric policies is also supported.

![](./docs/img/gifs/fabrics_collage.gif)

[comment]: < See the documentation for a complete description, [Fabrics Sim Documentation](https://srl.gitlab-master-pages.nvidia.com/fabrics_sim/).>

## Installation
1. [Install Isaac Sim 4.5](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html) via the Omniverse launcher and ensure Isaac Sim runs 

2. [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

3. [Create Isaac Sim Conda environment](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_python.html#advanced-running-with-anaconda) where `environment.yaml` is within the top level Isaac Sim install directory, e.g., `/home/<user>/.local/share/ov/pkg/isaac-sim-*`. The dependencies within `environment.yml` can also be installed manually via pip within your Isaac Sim conda environment.

4. Install poetry and install project

      curl -sSL https://install.python-poetry.org | python3 -
      cd <fabrics_dir>
      poetry install

5. Patch `urdfpy` dependency `networkx` to work with python 3.10

       cd <fabrics_dir>
       chmod +x urdfpy_patch.sh
       ./urdfpy_patch.sh

6. Deactivate and reactivate conda env (there persists a networkx issue otherwise)

       conda deactivate
       conda activate isaac-sim

7. Source Isaac Sim's conda setup bash script, where `isaac_sim_path` is something like `/home/<user>/.local/share/ov/pkg/isaac-sim-*`

       source <isaac_sim_path>/setup_conda_env.sh



## Examples
Once installed, you should be able to run the example scripts in the example directory, e.g.:

Kuka-Allegro (DextrAH) fabric:

    python <fabrics_dir>/examples/kuka_allegro_pose_fabric_example.py --batch_size=10 --render --cuda_graph

## Notes
One can update dependences in deps.txt file, remove pyproject.toml and poetry.lock files, and regenerate them

    cd <fabrics_dir>
    rm pyproject.toml poetry.lock
    poetry init --no-interaction
    xargs poetry add < deps.txt
    poetry install
