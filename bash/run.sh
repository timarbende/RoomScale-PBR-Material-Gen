stamp=$(date "+%Y-%m-%d_%H-%M-%S")

checkpoint_dir="."   # training with sds, guidance = 100.0
conditioning_texture_step="20000"  # which first stage output step's result texture to be used
desired_aov="albedo"
log_dir="outputs/${desired_aov}"
prompt="a sci-fi style basement with green floor"
scene_id="93f59740-4b65-4e8b-8a0f-6420b339469d/room_4"
python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --aov="$desired_aov" --conditioning_texture_step "$conditioning_texture_step"