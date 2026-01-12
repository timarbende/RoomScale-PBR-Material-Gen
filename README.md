# Room-Scale Image-Conditional PBR Material Generation

Recent developments in Score Distillation Sampling led to pipelines generating high-quality, visually plausible 3D assets for virtual environments with image and text conditioning. The primary limitation of these approaches is that they do not separate the material from the light, resulting in textures with baked-in lighting effects. In contrast, we proposed a method that utilizes a pre-trained monocular generative intrinsic image estimator to distill Physically Based Rendering material textures for room scale scenes. Given the scene geometric mesh and a set of images spatially covering the environment, our model synthesizes albedo, roughness, metallic, normal and irradiance maps using score distillation. Our method outperforms state-of-the-art optimization-based methods and provides clean material maps, which can subsequently be used in rendering engines for the purpose of relighting or material editing.

All experiments were launched on a workstation equipped with a single NVIDIA GeForce RTX 3090 GPU with 24GB VRAM available, running CUDA Version 12.9. A floating-point
precision of 32-bits was used, as 16-bit precision results in gradient explosion and oversaturated model outputs. Average runtime of generating a single material map was found to be 4.5 hours with 17K training iterations.

Setup and run instructions are to be added soon
