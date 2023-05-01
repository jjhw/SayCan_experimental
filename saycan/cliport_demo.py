from main import *
def run_cliport(env, obs, text, coords, clip_model, optim):
    before = env.get_camera_image()
    prev_obs = obs['image'].copy()

    # Tokenize text and get CLIP features.
    text_tokens = clip.tokenize(text).cuda()
    with torch.no_grad():
        text_feats = clip_model.encode_text(text_tokens).float()
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    text_feats = np.float32(text_feats.cpu())

    # Normalize image and add batch dimension.
    img = obs['image'][None, ...] / 255
    # show img
    img = np.concatenate((img, coords[None, ...]), axis=3)

    # Run Transporter Nets to get pick and place heatmaps.
    batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
    pick_map, place_map = eval_step(optim.target, batch)
    pick_map, place_map = np.float32(pick_map), np.float32(place_map)

    # Get pick position.
    pick_max = np.argmax(np.float32(pick_map)).squeeze()
    pick_yx = (pick_max // 224, pick_max % 224)
    pick_yx = np.clip(pick_yx, 20, 204)
    pick_xyz = obs['xyzmap'][pick_yx[0], pick_yx[1]]

    # Get place position.
    place_max = np.argmax(np.float32(place_map)).squeeze()
    place_yx = (place_max // 224, place_max % 224)
    place_yx = np.clip(place_yx, 20, 204)
    place_xyz = obs['xyzmap'][place_yx[0], place_yx[1]]

    # Step environment.
    act = {'pick': pick_xyz, 'place': place_xyz}
    obs, _, _, _ = env.step(act)

    # Show pick and place action.
    plt.title(text)
    plt.imshow(prev_obs)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()

    # Show debug plots.
    plt.subplot(1, 2, 1)
    plt.title('Pick Heatmap')
    plt.imshow(pick_map.reshape(224, 224))
    plt.subplot(1, 2, 2)
    plt.title('Place Heatmap')
    plt.imshow(place_map.reshape(224, 224))
    plt.show()

    # Show video of environment rollout.
    # debug_clip = ImageSequenceClip(env.cache_video, fps=25)
    #   display(debug_clip.ipython_display(autoplay=1, loop=1, center=False))
    env.cache_video = []

    # Show camera image after pick and place.
    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(before)
    plt.subplot(1, 2, 2)
    plt.title('After')
    after = env.get_camera_image()
    plt.imshow(after)
    plt.show()

    # return pick_xyz, place_xyz, pick_map, place_map, pick_yx, place_yx
    return obs

display_input_size = (10, 10)
overall_fig_size = (18, 24)


env = PickPlaceEnv()
# Load CLIP
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.cuda().eval()
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
print("Input resolution:", clip_model.visual.input_resolution)
print("Context length:", clip_model.context_length)
print("Vocab size:", clip_model.vocab_size)


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
saved_model_dir = "/workspace/saycan/weights/image_path_v2"
_ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

numbered_categories = [{"name": str(idx), "id": idx,} for idx in range(50)]
numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}
image_path = 'tmp.jpg'


config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)
img = env.get_camera_image()
plt.imshow(img)
plt.show()
# Coordinate map (i.e. position encoding).
coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)
user_input = 'Pick the yellow block and place it on the blue bowl.'  #@param {type:"string"}
# Initialize model weights using dummy tensors.
rng = jax.random.PRNGKey(0)
rng, key = jax.random.split(rng)
init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
init_text = jnp.ones((4, 512), jnp.float32)
init_pix = jnp.zeros((4, 2), np.int32)
init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
obs = run_cliport(env = env, obs = obs, text = user_input, coords = coords, 
                    clip_model = clip_model, optim = optim)
