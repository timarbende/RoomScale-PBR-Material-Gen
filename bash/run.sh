stamp=$(date "+%Y-%m-%d_%H-%M-%S")

checkpoint_dir="."   # training with sds, guidance = 100.0
checkpoint_step="29900"  # explicitly set to last checkpoint of training
log_dir="outputs/"
prompt="a sci-fi style basement with green floor"
scene_id="93f59740-4b65-4e8b-8a0f-6420b339469d/room_4"
python scripts/train_texture.py --config config/template.yaml --stamp $stamp --log_dir $log_dir --prompt "$prompt" --scene_id "$scene_id" --checkpoint_dir "$checkpoint_dir" --checkpoint_step "$checkpoint_step"