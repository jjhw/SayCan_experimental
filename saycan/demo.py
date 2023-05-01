from main import *

display_input_size = (10, 10)
overall_fig_size = (18, 24)


env = PickPlaceEnv()
# config = {'pick':  ['yellow block', 'green block', 'blue block'],
#           'place': ['yellow bowl', 'green bowl', 'blue bowl']}

# np.random.seed(42)
# obs = env.reset(config)
# img = env.get_camera_image_top()
# img = np.flipud(img.transpose(1, 0, 2))
# plt.title('ViLD Input Image')
# plt.imshow(img)
# plt.show()
# imageio.imwrite('tmp.jpg', img)

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

if False:
    #@markdown ViLD settings.
    category_names = ['blue block',
                    'red block',
                    'green block',
                    'orange block',
                    'yellow block',
                    'purple block',
                    'pink block',
                    'cyan block',
                    'brown block',
                    'gray block',

                    'blue bowl',
                    'red bowl',
                    'green bowl',
                    'orange bowl',
                    'yellow bowl',
                    'purple bowl',
                    'pink bowl',
                    'cyan bowl',
                    'brown bowl',
                    'gray bowl']
    category_name_string = ";".join(category_names)
    max_boxes_to_draw = 8 #@param {type:"integer"}

    # Extra prompt engineering: swap A with B for every (A, B) in list.
    prompt_swaps = [('block', 'cube')]

    nms_threshold = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
    max_box_area = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
    vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
    vild_obj = VILD(session, overall_fig_size, numbered_category_indices, clip_model)
    found_objects = vild_obj.vild(image_path, category_name_string, vild_params, plot_on=True, prompt_swaps=prompt_swaps)
    print(found_objects)

if False:
    # Show dataset example
    dataset, dataset_size = dataset_load(True)
    img = dataset['image'][0]
    pick_yx = dataset['pick_yx'][0]
    place_yx = dataset['place_yx'][0]
    text = dataset['text'][0]
    plt.title(text)
    plt.imshow(img)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()

# #@markdown Compute CLIP features for text in the dataset.

# # Precompute CLIP features for all text in training dataset.
# text_tokens = clip.tokenize(dataset['text']).cuda()
# text_i = 0
# data_text_feats = np.zeros((0, 512), dtype=np.float32)
# while text_i < len(text_tokens):
#   batch_size = min(len(text_tokens) - text_i, 512)
#   text_batch = text_tokens[text_i:text_i+batch_size]
#   with torch.no_grad():
#     batch_feats = clip_model.encode_text(text_batch).float()
#   batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
#   batch_feats = np.float32(batch_feats.cpu())
#   data_text_feats = np.concatenate((data_text_feats, batch_feats), axis=0)
#   text_i += batch_size

config = {'pick':  ['yellow block', 'green block', 'blue block'],
          'place': ['yellow bowl', 'green bowl', 'blue bowl']}

np.random.seed(42)
obs = env.reset(config)
img = env.get_camera_image()
# plt.imshow(img)
# plt.show()
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
