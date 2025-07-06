stamp=$(date "+%Y-%m-%d_%H-%M-%S")

checkpoint_dir="outputs/a_bohemian_style_living_room/sds/2025-06-26_16-34-01"   # run with sds, guidance = 100.0
log_dir="outputs/" # TODO
prompt="a sci-fi style living room with green floor"
scene_id="93f59740-4b65-4e8b-8a0f-6420b339469d/room_4"
python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --checkpoint_dir "$checkpoint_dir"