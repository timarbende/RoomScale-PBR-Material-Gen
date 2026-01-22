def get_render_size(render_size):
    if render_size.__class__ is str:
        s = render_size.strip().strip("()")
        parts = s.split(",")
        return (int(parts[0]), int(parts[1]))
    else:
        return render_size