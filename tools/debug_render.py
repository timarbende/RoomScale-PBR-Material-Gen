import os
from models.modules import TextureMesh, Studio
import numpy as np
import torchvision
from torchvision.utils import save_image
import torch
from tools.tuple_reader import get_render_size
from tools.exr_utils import save_tensor_to_exr
from PIL import Image
from pytorch3d.renderer import TexturesUV

from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.set_device(DEVICE)
else:
    print("no gpu avaiable")
    exit()

def debug_resize(photo):
    old_height = photo.shape[1]
    old_width = photo.shape[2]
    new_height = old_height
    new_width = old_width
    radio = old_height / old_width
    max_side = 1000
    if old_height > old_width:
        new_height = max_side
        new_width = int(new_height / radio)
    else:
        new_width = max_side
        new_height = int(new_width * radio)

    if new_width % 8 != 0 or new_height % 8 != 0:
        new_width = new_width // 8 * 8
        new_height = new_height // 8 * 8

    resized_photo = torchvision.transforms.Resize((new_height, new_width))(photo)
    torchvision.transforms.ToPILImage()(resized_photo).save("resized.png")

def debug_render_all_views(config):
    cameras_count = 10
    studio = Studio(config, DEVICE)
    texture_mesh = TextureMesh(config, DEVICE)

    for camera_id in range(cameras_count):
        Rs, Ts, fovs, _, image_path = studio.sample_cameras(camera_id, config.batch_size, random_cameras=False)            
        camera = studio.set_cameras(config.camera_type, Rs, Ts, fovs)

        renderer = studio.set_renderer(camera, get_render_size(config.render_size))
        _, fragments = studio.render(renderer, texture_mesh.mesh, texture_mesh.texture, None, None, None, True)
        _, relative_depth = studio.get_relative_depth_map(fragments.zbuf)
        relative_depth = relative_depth / 255.0

        image_path = image_path.split("/")[-1]

        save_image(relative_depth, "{}.png".format(image_path))

def debug_render(config, aov):
    print("rendering {}".format(aov))

    directory = os.path.join("comparisons", config.wandb_run_name)

    config.aov = aov

    material_path = os.path.join(directory, "{}_inpainted.png".format(aov))
    studio = Studio(config, DEVICE)
    texture_mesh = TextureMesh(config, DEVICE)
    
    texture_img = Image.open(material_path)
    texture = torchvision.transforms.ToTensor()(texture_img).permute(1, 2, 0).cuda()

    texture_mesh.mesh.textures = TexturesUV(
        maps=texture[None, ...],  # B, H, W, C
        faces_uvs= texture_mesh.mesh.textures.faces_uvs_padded(),
        verts_uvs= texture_mesh.mesh.textures.verts_uvs_padded(),
        sampling_mode="bilinear"
    )

    output_path = os.path.join(directory, aov)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for camera_id, _ in enumerate(tqdm(range(studio.num_cameras))):
        Rs, Ts, fovs, _, image_path = studio.sample_cameras(camera_id, config.batch_size, random_cameras=False)            
        camera = studio.set_cameras(config.camera_type, Rs, Ts, fovs)

        renderer = studio.set_renderer(camera, get_render_size(config.render_size))
        features, _ = studio.render(renderer, texture_mesh.mesh, texture_mesh.texture, None, None, None, True)
        features = features.squeeze(0).permute(2, 0, 1)[:3]
        if(config.aov == "roughness" or config.aov == "metallic"):  # on roughness and metallic only take first channel (rgbx convention)
            features = features[0:1].repeat(3, 1, 1)

        image_path = image_path.split("/")[-1]

        save_tensor_to_exr(features, os.path.join(output_path, "{}.exr".format(image_path)))

def debug_render_latent(config):
    config.aov = ""

    studio = Studio(config, DEVICE)
    texture_mesh = TextureMesh(config, DEVICE)
    
    texture_img = Image.open("kitchen_latent_texture.png")
    texture = torchvision.transforms.ToTensor()(texture_img).permute(1, 2, 0).cuda()

    texture_mesh.mesh.textures = TexturesUV(
        maps=texture[None, ...],  # B, H, W, C
        faces_uvs= texture_mesh.mesh.textures.faces_uvs_padded(),
        verts_uvs= texture_mesh.mesh.textures.verts_uvs_padded(),
        sampling_mode="bilinear"
    )

    for camera_id, _ in enumerate(tqdm(range(10))):
        Rs, Ts, fovs, _, _ = studio.sample_cameras(camera_id, config.batch_size, random_cameras=False)            
        camera = studio.set_cameras(config.camera_type, Rs, Ts, fovs)

        renderer = studio.set_renderer(camera, get_render_size(config.render_size))
        features, _ = studio.render(renderer, texture_mesh.mesh, texture_mesh.texture, None, None, None, True)
        features = features.squeeze(0).permute(2, 0, 1)[:3]

        save_image(features, "{}.png".format(camera_id))

def debug_render_depth(config):
    cameras_count = 10
    studio = Studio(config, DEVICE)
    texture_mesh = TextureMesh(config, DEVICE)

    for camera_id in range(cameras_count):
        Rs, Ts, fovs, _, image_path = studio.sample_cameras(camera_id, config.batch_size, random_cameras=False)            
        camera = studio.set_cameras(config.camera_type, Rs, Ts, fovs)

        renderer = studio.set_renderer(camera, config.render_size)
        _, fragments = studio.render(renderer, texture_mesh.mesh, texture_mesh.texture, None, None, None, True)
        _, relative_depth = studio.get_relative_depth_map(fragments.zbuf)
        relative_depth = relative_depth / 255.0

        image_path = image_path.split("/")[-1]

        save_image(relative_depth, "{}.png".format(image_path))

def debug_render_texture(config):
    texture_mesh = TextureMesh(config, DEVICE)

    decoded_texture_not_normalized = texture_mesh.texture[0].permute(2, 0, 1)
    torchvision.transforms.ToPILImage()(decoded_texture_not_normalized).save("original_texture.png")
    decoded_texture = (decoded_texture_not_normalized / 2 + 0.5).clamp(0, 1)
    decoded_texture = torchvision.transforms.ToPILImage()(decoded_texture).convert("RGB")
    decoded_texture.save("debug_texture.png")
