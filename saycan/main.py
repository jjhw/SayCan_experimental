#! /usr/bin/env python3

import collections
import datetime
import os
import random
import threading
import time

import cv2  # Used by ViLD.
import clip
import flax
from flax import linen as nn
from flax.training import checkpoints
from flax.metrics import tensorboard
import imageio
from heapq import nlargest
import jax
import jax.numpy as jnp
# from moviepy.editor import ImageSequenceClip
import numpy as np
import openai
import optax
import pickle
import pybullet
import pybullet_data
import tensorflow.compat.v1 as tf
import torch
from tqdm import tqdm
import os
import logging
from jax.lib import xla_bridge
import matplotlib.pyplot as plt

from gymenv import PickPlaceEnv
from vild_main import VILD
from oracle import ScriptedPolicy
from defines import PICK_TARGETS, PLACE_TARGETS
from model import TransporterNets, n_params
from gpt3 import GPT3

logger = logging.getLogger("saycan.main")

try:
    openai_api_key = os.environ['OPENAI_KEY']
except KeyError:
    logger.fatal('environment variable "OPENAI_KEY" not found')
    exit(-1)

openai.api_key = openai_api_key
logger.info(f'using platform: {xla_bridge.get_backend().platform}')

def dataset_load(load_pregenerated):

    # Load pre-existing dataset.
    if load_pregenerated:
        dataset = pickle.load(open('/workspace/saycan/datasets/dataset-9999.pkl', 'rb'))  # ~10K samples.
        dataset_size = len(dataset['text'])

    # Generate new dataset.
    else:
        dataset = {}
        dataset_size = 2  # Size of new dataset.
        dataset['image'] = np.zeros((dataset_size, 224, 224, 3), dtype=np.uint8)
        dataset['pick_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
        dataset['place_yx'] = np.zeros((dataset_size, 2), dtype=np.int32)
        dataset['text'] = []
        policy = ScriptedPolicy(env)
        data_idx = 0
        while data_idx < dataset_size:
            np.random.seed(data_idx)
            num_pick, num_place = 3, 3

            # Select random objects for data collection.
            pick_items = list(PICK_TARGETS.keys())
            pick_items = np.random.choice(pick_items, size=num_pick, replace=False)
            place_items = list(PLACE_TARGETS.keys())
            for pick_item in pick_items:  # For simplicity: place items != pick items.
                place_items.remove(pick_item)
            place_items = np.random.choice(place_items, size=num_place, replace=False)
            config = {'pick': pick_items, 'place': place_items}

            # Initialize environment with selected objects.
            obs = env.reset(config)

            # Create text prompts.
            prompts = []
            for i in range(len(pick_items)):
                pick_item = pick_items[i]
                place_item = place_items[i]
                prompts.append(f'Pick the {pick_item} and place it on the {place_item}.')

            # Execute 3 pick and place actions.
            for prompt in prompts:
                act = policy.step(prompt, obs)
                dataset['text'].append(prompt)
                dataset['image'][data_idx, ...] = obs['image'].copy()
                dataset['pick_yx'][data_idx, ...] = xyz_to_pix(act['pick'])
                dataset['place_yx'][data_idx, ...] = xyz_to_pix(act['place'])
                data_idx += 1
                obs, _, _, _ = env.step(act)
                # debug_clip = ImageSequenceClip(env.cache_video, fps=25)
                # display(debug_clip.ipython_display(autoplay=1, loop=1))
                env.cache_video = []
                if data_idx >= dataset_size:
                    break

    pickle.dump(dataset, open(f'dataset-{dataset_size}.pkl', 'wb'))
    return dataset, dataset_size



@jax.jit
def train_step(optimizer, batch):
  def loss_fn(params):
    batch_size = batch['img'].shape[0]
    pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'], batch['pick_yx'])

    # InfoNCE pick loss.
    pick_logits = pick_logits.reshape(batch_size, -1)
    pick_onehot = batch['pick_onehot'].reshape(batch_size, -1)
    pick_loss = jnp.mean(optax.softmax_cross_entropy(logits=pick_logits, labels=pick_onehot), axis=0)

    # InfoNCE place loss.
    place_logits = place_logits.reshape(batch_size, -1)
    place_onehot = batch['place_onehot'].reshape(batch_size, -1)
    place_loss = jnp.mean(optax.softmax_cross_entropy(logits=place_logits, labels=place_onehot), axis=0)
    
    loss = pick_loss + place_loss
    return loss, (pick_logits, place_logits)
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grad = grad_fn(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss, grad, logits

@jax.jit
def eval_step(params, batch):
  pick_logits, place_logits = TransporterNets().apply({'params': params}, batch['img'], batch['text'])
  return pick_logits, place_logits


#@title Helper Functions

def build_scene_description(found_objects, block_name="box", bowl_name="circle"):
  scene_description = f"objects = {found_objects}"
  scene_description = scene_description.replace(block_name, "block")
  scene_description = scene_description.replace(bowl_name, "bowl")
  scene_description = scene_description.replace("'", "")
  return scene_description

def step_to_nlp(step):
  step = step.replace("robot.pick_and_place(", "")
  step = step.replace(")", "")
  pick, place = step.split(", ")
  return "Pick the " + pick + " and place it on the " + place + "."

def normalize_scores(scores):
  max_score = max(scores.values())  
  normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
  return normed_scores

def plot_saycan(llm_scores, vfs, combined_scores, task, correct=True, show_top=None):
  if show_top:
    top_options = nlargest(show_top, combined_scores, key = combined_scores.get)
    # add a few top llm options in if not already shown
    top_llm_options = nlargest(show_top // 2, llm_scores, key = llm_scores.get)
    for llm_option in top_llm_options:
      if not llm_option in top_options:
        top_options.append(llm_option)
    llm_scores = {option: llm_scores[option] for option in top_options}
    vfs = {option: vfs[option] for option in top_options}
    combined_scores = {option: combined_scores[option] for option in top_options}

  sorted_keys = dict(sorted(combined_scores.items()))
  keys = [key for key in sorted_keys]
  positions = np.arange(len(combined_scores.items()))
  width = 0.3

  fig = plt.figure(figsize=(12, 6))
  ax1 = fig.add_subplot(1,1,1)

  plot_llm_scores = normalize_scores({key: np.exp(llm_scores[key]) for key in sorted_keys})
  plot_llm_scores = np.asarray([plot_llm_scores[key] for key in sorted_keys])
  plot_affordance_scores = np.asarray([vfs[key] for key in sorted_keys])
  plot_combined_scores = np.asarray([combined_scores[key] for key in sorted_keys])
  
  ax1.bar(positions, plot_combined_scores, 3 * width, alpha=0.6, color="#93CE8E", label="combined")
    
  score_colors = ["#ea9999ff" for score in plot_affordance_scores]
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#ea9999ff", label="vfs")
  ax1.bar(positions + width / 2, 0 * plot_combined_scores, width, color="#a4c2f4ff", label="language")
  ax1.bar(positions - width / 2, np.abs(plot_affordance_scores), width, color=score_colors)
  
  plt.xticks(rotation="vertical")
  ax1.set_ylim(0.0, 1.0)

  ax1.grid(True, which="both")
  ax1.axis("on")

  ax1_llm = ax1.twinx()
  ax1_llm.bar(positions + width / 2, plot_llm_scores, width, color="#a4c2f4ff", label="language")
  ax1_llm.set_ylim(0.01, 1.0)
  plt.yscale("log")
  
  font = {"fontname":"Arial", "size":"16", "color":"k" if correct else "r"}
  plt.title(task, **font)
  key_strings = [key.replace("robot.pick_and_place", "").replace(", ", " to ").replace("(", "").replace(")","") for key in keys]
  plt.xticks(positions, key_strings, **font)
  ax1.legend()
  plt.show()

def affordance_scoring(options, found_objects, verbose=False, block_name="box", bowl_name="circle", termination_string="done()"):
  affordance_scores = {}
  found_objects = [
                   found_object.replace(block_name, "block").replace(bowl_name, "bowl") 
                   for found_object in found_objects + list(PLACE_TARGETS.keys())[-5:]]
  verbose and print("found_objects", found_objects)
  for option in options:
    if option == termination_string:
      affordance_scores[option] = 0.2
      continue
    pick, place = option.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
    affordance = 0
    found_objects_copy = found_objects.copy()
    if pick in found_objects_copy:
      found_objects_copy.remove(pick)
      if place in found_objects_copy:
        affordance = 1
    affordance_scores[option] = affordance
    verbose and print(affordance, '\t', option)
  return affordance_scores


if __name__ == "__main__":
    # initialize environment
    env = PickPlaceEnv()
    config = {'pick':  ['yellow block', 'green block', 'blue block'],
            'place': ['yellow bowl', 'green bowl', 'blue bowl']}
    np.random.seed(42)
    obs = env.reset(config)

    plt.subplot(1, 2, 1)
    img = env.get_camera_image()
    plt.title('Perspective side-view')
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    img = env.get_camera_image_top()
    img = np.flipud(img.transpose(1, 0, 2))
    plt.title('Orthographic top-view')
    plt.imshow(img)
    plt.show()

    # Note: orthographic cameras do not exist. But we can approximate them by
    # projecting a 3D point cloud from an RGB-D camera, then unprojecting that onto
    # an orthographic plane. Orthographic views are useful for spatial action maps.
    plt.title('Unprojected orthographic top-view')
    plt.imshow(obs['image'])
    plt.show()

    # end of environment initialization
    # VILD Demo

  
    np.random.seed(42)
    obs = env.reset(config)
    img = env.get_camera_image_top()
    img = np.flipud(img.transpose(1, 0, 2))
    plt.title('ViLD Input Image')
    plt.imshow(img)
    plt.show()
    imageio.imwrite('tmp.jpg', img)
    # End of VILD Demo
    
    clip_model, clip_preprocess = clip.load("ViT-B/32")
    clip_model.cuda().eval()
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("Input resolution:", clip_model.visual.input_resolution)
    print("Context length:", clip_model.context_length)
    print("Vocab size:", clip_model.vocab_size)
    #@markdown Define ViLD hyperparameters.
    


    # # Global matplotlib settings
    # SMALL_SIZE = 16#10
    # MEDIUM_SIZE = 18#12
    # BIGGER_SIZE = 20#14

    # plt.rc('font', size=MEDIUM_SIZE)         # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    # Parameters for drawing figure.
    display_input_size = (10, 10)
    overall_fig_size = (18, 24)

    line_thickness = 1
    fig_size_w = 35
    # fig_size_h = min(max(5, int(len(category_names) / 2.5) ), 10)
    mask_color =   'red'
    alpha = 0.5


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
    saved_model_dir = "/workspace/saycan/weights/image_path_v2"
    _ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)

    numbered_categories = [{"name": str(idx), "id": idx,} for idx in range(50)]
    numbered_category_indices = {cat["id"]: cat for cat in numbered_categories}





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
    image_path = 'tmp.jpg'

    #@markdown ViLD settings.
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
    image_path = 'tmp.jpg'

    #@markdown ViLD settings.
    category_name_string = ";".join(category_names)
    max_boxes_to_draw = 8 #@param {type:"integer"}

    # Extra prompt engineering: swap A with B for every (A, B) in list.
    prompt_swaps = [('block', 'cube')]

    nms_threshold = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
    max_box_area = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
    vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
    found_objects = vild_obj.vild(image_path, category_name_string, vild_params, plot_on=True, prompt_swaps=prompt_swaps)
    

    dataset, dataset_size = dataset_load(True)

    img = dataset['image'][0]
    pick_yx = dataset['pick_yx'][0]
    place_yx = dataset['place_yx'][0]
    text = dataset['text'][0]
    plt.title(text)
    plt.imshow(img)
    plt.arrow(pick_yx[1], pick_yx[0], place_yx[1]-pick_yx[1], place_yx[0]-pick_yx[0], color='w', head_starts_at_zero=False, head_width=7, length_includes_head=True)
    plt.show()






    
    
    #@markdown Compute CLIP features for text in the dataset.

    # Precompute CLIP features for all text in training dataset.
    text_tokens = clip.tokenize(dataset['text']).cuda()
    text_i = 0
    data_text_feats = np.zeros((0, 512), dtype=np.float32)
    while text_i < len(text_tokens):
        batch_size = min(len(text_tokens) - text_i, 512)
        text_batch = text_tokens[text_i:text_i+batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        data_text_feats = np.concatenate((data_text_feats, batch_feats), axis=0)
        text_i += batch_size

    # Coordinate map (i.e. position encoding).
    coord_x, coord_y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224), sparse=False, indexing='ij')
    coords = np.concatenate((coord_x[..., None], coord_y[..., None]), axis=2)



    name = datetime.datetime.now().strftime(f'%Y-%m-%d-%H:%M:%S-cliport')
    logdir = os.path.join("logs", name)
    writer = tensorboard.SummaryWriter(logdir)
    load_pretrained = True  #@param {type:"boolean"}

    # Initialize model weights using dummy tensors.
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    print(f'Model parameters: {n_params(init_params):,}')
    # optim = optax.adam(learning_rate=1e-4).create(init_params)
    # optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
    optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)
    # AttributeError: module 'flax' has no attribute 'optim'
    
    # Define and reset environment.
    config = {'pick':  ['yellow block', 'green block', 'blue block'],
            'place': ['yellow bowl', 'green bowl', 'blue bowl']}

    np.random.seed(42)
    obs = env.reset(config)
    img = env.get_camera_image()
    plt.imshow(img)
    plt.show()
    user_input = 'Pick the yellow block and place it on the blue bowl.'  #@param {type:"string"}
    ENGINE = "text-davinci-002"
    obs = run_cliport(obs, user_input, coords)
    gpt3 = GPT3(openai_api_key)
    query = "To pick the blue block and put it on the red block, I should:\n"
    options = gpt3.make_options(PICK_TARGETS, PLACE_TARGETS)
    scores, response = gpt3.gpt3_scoring(query, options, engine=ENGINE, limit_num_options=5, option_start='\n', verbose=True)

    termination_string = "done()"
    query = "To pick the blue block and put it on the red block, I should:\n"

    options = gpt3.make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
    llm_scores, _ = gpt3.gpt3_scoring(query, options, verbose=True, engine=ENGINE)

    affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False, termination_string=termination_string)

    combined_scores = {option: np.exp(llm_scores[option]) * affordance_scores[option] for option in options}
    combined_scores = normalize_scores(combined_scores)
    selected_task = max(combined_scores, key=combined_scores.get)
    print("Selecting: ", selected_task)
    termination_string = "done()"

    gpt3_context = """
objects = [red block, yellow block, blue block, green bowl]
# move all the blocks to the top left corner.
robot.pick_and_place(blue block, top left corner)
robot.pick_and_place(red block, top left corner)
robot.pick_and_place(yellow block, top left corner)
done()

objects = [red block, yellow block, blue block, green bowl]
# put the yellow one the green thing.
robot.pick_and_place(yellow block, green bowl)
done()

objects = [yellow block, blue block, red block]
# move the light colored block to the middle.
robot.pick_and_place(yellow block, middle)
done()

objects = [blue block, green bowl, red block, yellow bowl, green block]
# stack the blocks.
robot.pick_and_place(green block, blue block)
robot.pick_and_place(red block, green block)
done()

objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
# group the blue objects together.
robot.pick_and_place(blue block, blue bowl)
done()

objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
# sort all the blocks into their matching color bowls.
robot.pick_and_place(green block, green bowl)
robot.pick_and_place(red block, red bowl)
robot.pick_and_place(yellow block, yellow bowl)
done()
"""
    use_environment_description = False
    gpt3_context_lines = gpt3_context.split("\n")
    gpt3_context_lines_keep = []
    for gpt3_context_line in gpt3_context_lines:
        if "objects =" in gpt3_context_line and not use_environment_description:
            continue
        gpt3_context_lines_keep.append(gpt3_context_line)

    gpt3_context = "\n".join(gpt3_context_lines_keep)
    print(gpt3_context)
    only_plan = False

    raw_input = "put all the blocks in different corners." 
    config = {"pick":  ["red block", "yellow block", "green block", "blue block"],
            "place": ["red bowl"]}
    
    # load_pretrained = True  #@param {type:"boolean"}

    # # Initialize model weights using dummy tensors.
    # rng = jax.random.PRNGKey(0)
    # rng, key = jax.random.split(rng)
    # init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    # init_text = jnp.ones((4, 512), jnp.float32)
    # init_pix = jnp.zeros((4, 2), np.int32)
    # init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
    # print(f'Model parameters: {n_params(init_params):,}')
    # optim = flax.optim.Adam(learning_rate=1e-4).create(init_params)

    # if load_pretrained:
    #     ckpt_path = f'ckpt_{40000}'
    #     optim = checkpoints.restore_checkpoint(ckpt_path, optim)
    #     print('Loaded:', ckpt_path)
    # else:

    #     # Training loop.
    #     batch_size = 8
    #     for train_iter in range(1, 40001):
    #         batch_i = np.random.randint(dataset_size, size=batch_size)
    #         text_feat = data_text_feats[batch_i, ...]
    #         img = dataset['image'][batch_i, ...] / 255
    #         img = np.concatenate((img, np.broadcast_to(coords[None, ...], (batch_size,) + coords.shape)), axis=3)

    #         # Get onehot label maps.
    #         pick_yx = np.zeros((batch_size, 2), dtype=np.int32)
    #         pick_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
    #         place_onehot = np.zeros((batch_size, 224, 224), dtype=np.float32)
    #         for i in range(len(batch_i)):
    #             pick_y, pick_x  = dataset['pick_yx'][batch_i[i], :]
    #             place_y, place_x = dataset['place_yx'][batch_i[i], :]
    #             pick_onehot[i, pick_y, pick_x] = 1
    #             place_onehot[i, place_y, place_x] = 1
    #             # pick_onehot[i, ...] = scipy.ndimage.gaussian_filter(pick_onehot[i, ...], sigma=3)

    #             # Data augmentation (random translation).
    #             roll_y, roll_x = np.random.randint(-112, 112, size=2)
    #             img[i, ...] = np.roll(img[i, ...], roll_y, axis=0)
    #             img[i, ...] = np.roll(img[i, ...], roll_x, axis=1)
    #             pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_y, axis=0)
    #             pick_onehot[i, ...] = np.roll(pick_onehot[i, ...], roll_x, axis=1)
    #             place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_y, axis=0)
    #             place_onehot[i, ...] = np.roll(place_onehot[i, ...], roll_x, axis=1)
    #             pick_yx[i, 0] = pick_y + roll_y
    #             pick_yx[i, 1] = pick_x + roll_x

    #         # Backpropagate.
    #         batch = {}
    #         batch['img'] = jnp.float32(img)
    #         batch['text'] = jnp.float32(text_feat)
    #         batch['pick_yx'] = jnp.int32(pick_yx)
    #         batch['pick_onehot'] = jnp.float32(pick_onehot)
    #         batch['place_onehot'] = jnp.float32(place_onehot)
    #         rng, batch['rng'] = jax.random.split(rng)
    #         optim, loss, _, _ = train_step(optim, batch)
    #         writer.scalar('train/loss', loss, train_iter)

    #         if train_iter % np.power(10, min(4, np.floor(np.log10(train_iter)))) == 0:
    #             print(f'Train Step: {train_iter} Loss: {loss}')
            
    #         if train_iter % 1000 == 0:
    #             checkpoints.save_checkpoint('.', optim, train_iter, prefix='ckpt_', keep=100000, overwrite=True)