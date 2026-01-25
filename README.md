# Room-Scale Image-Conditional PBR Material Generation

<img width="841" height="307" alt="scannet_bedroom_relighting_results" src="https://github.com/user-attachments/assets/378e5ae9-d726-4aae-9b27-ba0e042b16e7" />

This is the project of my Master's Thesis, developed in the Visual Computing Chair of the Technical University of Munich.

Recent developments in Score Distillation Sampling led to pipelines generating high-quality, visually plausible 3D assets for virtual environments with image and text conditioning. The primary limitation of these approaches is that they do not separate the material from the light, resulting in textures with baked-in lighting effects. In contrast, we proposed a method that utilizes a pre-trained monocular generative intrinsic image estimator to distill Physically Based Rendering material textures for room scale scenes. Given the scene geometric mesh and a set of images spatially covering the environment, our model synthesizes albedo, roughness, metallic, normal and irradiance maps using score distillation. Our method outperforms state-of-the-art optimization-based methods and provides clean material maps, which can subsequently be used in rendering engines for the purpose of relighting or material editing.

All experiments were launched on a workstation equipped with a single NVIDIA GeForce RTX 3090 GPU with 24GB VRAM available, running CUDA Version 12.9. A floating-point
precision of 32-bits was used, as 16-bit precision results in gradient explosion and oversaturated model outputs. Average runtime of generating a material map was found to be 4.5 hours with 17K training iterations.

Detailed information can be found in the `masters_thesis.pdf` file. Demo videos can be found in the `thesis_presentation.pptx` file.

## Setup

To create the conda environment and install required dependencies, run setup.sh from the repository root
```
./setup.sh
```
This will create a conda environment called **MatGen**. Activate the environment
```
conda activate MatGen
```

## Run 

Data is included in the *data* folder in separate `.zip` files for each pre-configured scene, e.g. `kitchen_hq.zip`. Unzip the data into the *data* folder.

To optimize PBR material for a scene, run one of the bash scripts in the *bash* folder
```
./bash/run_kitchen_hq.sh
```

Results are put into the *output/[scene_name]/[timestamp]/[PBR material parameter]_results* folder.

## Configuration

There are sample pipeline configuration files in the *config* folder, e.g. `kitchen_hq_config.yaml` to set pipeline parameters, output destination, logging methods and more.

## Custom scenes

To run the pipeline with custom data provided by you, first create a respective `.yaml` configuration file based on the ones provided in the *config* folder.
- scene geometry must be provided using `.obj` files containing UV texture coordinates, either as one combined scene mesh or as separate object meshes making up the scene accompanied by a scene config `.json` file. To use a combined scene mesh, set the `mesh_init` parameter to `scene_obj` and set the `scene_mesh_path` parameter to the object file. To use the object meshes and scene config file combination, set the `mesh_init` parameter to `scene_config` and set the `scene_config_path` parameter to the config file. You can find sample scene config files in the `3DFront_scenes.zip` file
- image captures covering the entire scene must be provided for best results
- camera poses and locations of respective image captures must be provided in a `.json` file. In case your data is from one of the datasets of the pre-configured scenes (3DFront, ScanNet++ or Bitterli's Rendering Resources), set the `camera_type` parameter as follows: 3DFront - `sphere` or `blender`,  ScanNet++ - `scannet`, Rendering Resources - `kitchen_hq` to run the correct pose normalization

### View-based weighting

The pipeline implements a view-count-based inverse weighting scheme to fix visual artifacts originating from differing geometry obversation counts. This is done by reading in the number of observations from a separate texture file. The pre-configured scenes are already equipped with this file. To generate the observation file for your own scene, run the view-count pipeline with the pipeline config file of your data
```
python scripts/train_view_count_texture.py --config config/[your_pipeline_config].yaml
```

You can also swap out the pipeline config file parameter in the `run_view_count.sh` file and run it.

## Acknowledgement

The project is a fork of Dave Zhenyu Chen et. al.'s [SceneTex pipeline](https://github.com/daveredrum/SceneTex), whom we would like to thank greatly. The reason for this repo not being a fork is to include the large volume of data in the *data* folder. We would further like to thank Zheng Zeng et. al. for [RGBX](https://github.com/zheng95z/rgbx) which was used as diffusion backbone.
