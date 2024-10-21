from collections import Counter
from itertools import chain
import logging
import multiprocessing
import os
from typing import Any
from time import time
import gc


import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
from termcolor import colored
import torch
import torch.distributed as dist
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs, evaluate_multitask_training_success
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)

# import cv2
# from pathlib import Path
# import sys
# sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from mode.evaluation.multistep_sequences import get_sequences
from mode.evaluation.utils import get_env_state_for_initial_condition, join_vis_lang, LangEmbeddings
from mode.rollout.rollout_video import RolloutVideo
from typing import Any, Dict, Tuple, Union


log_print = logging.getLogger(__name__)


def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return
    log_print.info(*args, **kwargs)


def divide_across_ranks(elements, world_size, rank):
    """
    Divide a number across subprocesses in multiprocessing.
    Example: distribute 4 elements in a world of size 3
    rank 0->2, rank 1->1, rank 2->1
    """
    assert rank < world_size
    rest = lambda n, w, i: 1 if n % w > i else 0
    return elements // world_size + rest(elements, world_size, rank)



def sequences_for_rank(num_sequences):
    """
    When using ddp, determine how many sequences every process should evaluate.
    """
    rank = dist.get_rank()
    ws = dist.get_world_size()
    num_seq_per_gpu = divide_across_ranks(num_sequences, ws, rank)
    num_workers = multiprocessing.cpu_count() // ws
    print(num_workers)
    print(num_seq_per_gpu)
    print(ws)
    print(rank)
    sequences = get_sequences(num_sequences, num_workers=num_workers)
    # print("Sequences:", sequences)

    print(f"Type of sequences: {type(sequences)}")
    print(f"Length of sequences: {len(sequences)}")
    print(f"First few elements of sequences: {sequences[:5]}")

    def manual_split(seq, n):
        avg = len(seq) // n
        remain = len(seq) % n
        last = 0
        results = []
        for _ in range(n):
            step = avg + (1 if remain > 0 else 0)
            results.append(seq[last:last+step])
            last += step
            remain -= 1
        return results

    try:
        sequences_np = np.array(sequences)
    except Exception as e:
        print(f"Exception when converting to numpy array: {e}")

    return manual_split(get_sequences(num_sequences, num_workers=num_workers), ws)[rank][:num_seq_per_gpu]


def gather_results(local_results):
    """
    Collect eval results from all processes and average them.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return local_results
    world_size = torch.distributed.get_world_size()
    print(f"Gathering results from {world_size} GPUs...")  # Logging the number of GPUs

    # Log local results
    print(f"Local results before gathering: {local_results}")

    results = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(results, local_results)

    # Check that all lists have the same length
    if not all(len(lst) == len(local_results) for lst in results):
        raise ValueError("All result lists must have the same length")

    # Calculate the average of each corresponding element
    averaged_results = [sum(values) / world_size for values in zip(*results)]

    # Log and return the averaged results
    print(f"Averaged results: {averaged_results}")
    return averaged_results


def get_video_tag(i):
    if dist.is_available() and dist.is_initialized():
        i = i * dist.get_world_size() + dist.get_rank()
    return f"_long_horizon/sequence_{i}"



class RolloutLibero(Callback):
    """
    A class for performing rollouts during validation step.
    """

    def __init__(
        self,
        benchmark_name,
        env_cfg,
        skip_epochs,
        rollout_freq,
        num_videos,
        num_sequences,
        max_steps,
        num_procs,
        n_eval,
        use_mp,
        task_embedding_format,
        empty_cache,
        debug,
        device,
    ):
        self.device = device
        self.task_order = 0
        self.bddl_folder = get_libero_path("bddl_files")
        self.init_states_folder = get_libero_path("init_states")
        self.task_embedding_format =task_embedding_format
        self.benchmark_name = benchmark_name
        self.benchmark_dict = benchmark.get_benchmark_dict()
        self.benchmark_instance = self.benchmark_dict[self.benchmark_name]()
        self.num_tasks = self.benchmark_instance.get_num_tasks()
        self.task_names = self.benchmark_instance.get_task_names()
        self.benchmark = get_benchmark(self.benchmark_name)(self.task_order)
        self.n_eval = n_eval
        self.env_cfg = env_cfg
        self.img_h = 224
        self.img_w = 224
        self.rank = None
        self.world_size = None
        self.skip_epochs = skip_epochs
        self.rollout_freq = rollout_freq
        self.num_videos = num_videos
        self.num_sequences = num_sequences
        self.max_steps = max_steps
        self.num_procs = num_procs
        self.use_mp = use_mp
        # self.save_dir = save_dir
        self.empty_cache = empty_cache
        self.debug = debug
        self.device = None
        self.eval_sequences = None
        self.init_states_paths = []
        self.cfg = {}
        self.descriptions = []
        self.create_cfg_for_libero(self.task_embedding_format)
        for i in range(self.num_tasks):

            task_i = self.benchmark_instance.get_task(0)

            self.init_states_paths.append(
                os.path.join(self.init_states_folder, self.task_names[i], task_i.init_states_file)
            )
            self.descriptions.append(self.benchmark_instance.get_task(i).language)
            task_embs = get_task_embs(self.cfg, self.descriptions)
            self.benchmark_instance.set_task_embs(task_embs)

        self.all_tasks = list(range(self.benchmark_instance.n_tasks))

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the validation loop begins."""
        if self.benchmark is None:
            self.device = pl_module.device

            if dist.is_available() and dist.is_initialized():
                self.eval_sequences = sequences_for_rank(self.num_sequences)
            else:
                self.eval_sequences = get_sequences(self.num_sequences)

            self.benchmark = get_benchmark(self.benchmark_name)(self.eval_sequences)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule, *args) -> None:

        rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = pl_module.device
        # self.dtype = pl_module.dtype
        transforms = trainer.datamodule.transforms['val']
        self.transforms = hydra.utils.instantiate(transforms)
        # get the

        if pl_module.current_epoch == 0 and self.skip_epochs > 0:
            # for i in range(self.num_tasks):
            #    pl_module.log(f"eval_lh/sr_chain_{i}", torch.tensor(0.0), on_step=False, sync_dist=True)
            pl_module.log("eval_lh/avg_seq_len", torch.tensor(0.0), on_step=False, sync_dist=True)
        elif pl_module.current_epoch == self.skip_epochs or ((pl_module.current_epoch - self.skip_epochs) >= 0 and (pl_module.current_epoch - self.skip_epochs) % self.rollout_freq == 0):
            successes = self.evaluate_policy(pl_module)

            successes = gather_results(successes)
            # if rank == 0:  # Only rank 0 performs the final aggregation
            result_array = sum(successes) / len(successes)

            # print(f"number of rollouts: {len(successes)}")
            log_rank_0(f"eval_lh/avg_seq_len success rate {torch.tensor(result_array)}")
            pl_module.log("eval_lh/avg_seq_len", torch.tensor(result_array), on_epoch=True, sync_dist=True)

            for success, task_name in zip(successes, self.task_names):
                log_rank_0(f"eval_lh/sr_{task_name} with success {success}")
                pl_module.log(f"eval_lh/sr_{task_name}", success, on_step=False, sync_dist=True)
            print('done')
            print()

    def evaluate_policy(self, model, store_video=False):
        # if not (dist.is_available() and dist.is_initialized()):
        #    raise RuntimeError("Distributed Data Parallel is not initialized.")

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        successes = []

        # Parallel evaluation across GPUs
        for idx in self.all_tasks:  # Distribute tasks across GPUs
            task_name = self.task_names[idx]
            task_i = self.benchmark_instance.get_task(idx)
            task_emb = self.benchmark_instance.task_embs[idx]
            task_str = f"k{self.all_tasks[-1]}_p{idx}"
            log_rank_0(f"GPU {rank} starting to evaluate: {task_name}")
            success_rate = self.evaluate_task_parallel(model, task_i, task_emb, task_str, idx, world_size, rank, store_video=store_video)
            successes.append(success_rate)

        return successes

        # Gather results from all GPUs
        # all_successes = gather_results(successes)
        # if rank == 0:  # Only rank 0 performs the final aggregation
            # if all_successes:
            #     aggregated_success_rate = sum(all_successes) / len(all_successes)
            # else:
            #    aggregated_success_rate = 0  # Handle the case where there are no successes
        #    return all_successes
        #else:
        #     return None  # Non-zero ranks do not return the aggregated results


    def evaluate_task_parallel(self, model, task_i, task_emb, task_str, idx, world_size, rank, sim_states=None, store_video=False):
        # Determine the number of rollouts per GPU
        rollouts_per_gpu = self.n_eval // world_size
        extra_rollouts = self.n_eval % world_size
        start_rollout = rank * rollouts_per_gpu + min(rank, extra_rollouts)
        end_rollout = start_rollout + rollouts_per_gpu + (1 if rank < extra_rollouts else 0)

        # Calculate the number of evaluations for this GPU
        eval_loop_num = end_rollout - start_rollout

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                self.bddl_folder, task_i.problem_folder, task_i.bddl_file
            ),
            "camera_heights": self.img_h,
            "camera_widths": self.img_w,
        }

        # Try to handle the frame buffer issue
        env_creation = False
        env_num = min(self.num_procs, rollouts_per_gpu) if self.use_mp else 1
        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            self.init_states_folder, task_i.problem_folder, task_i.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in tqdm(range(start_rollout, end_rollout), desc="Evaluating"):

            if store_video:
                video_frames = []
                video_filename = f"rollout_{task_str}_{rank}_{i}.mp4"
                video_path = os.path.join(store_video, video_filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec for MP4
                video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (self.img_w, self.img_h))

            env.reset()
            indices =  np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            # indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            model.reset()
            obs = env.set_init_state(init_states_)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i * env_num + k < self.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            while steps < self.max_steps:
                steps += 1
                data, goal = self.process_env_obs(obs[0], task_emb, task_i.language)
                # data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = model.step(data, goal).unsqueeze(0)
                actions = actions.cpu().numpy()
                obs, reward, done, info = env.step(actions)

                if store_video:
                    video_frames.append(obs[0]['agentview_image'])

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

                if all(dones):

                    break

            if store_video:
                for frame in video_frames:
                    video_writer.write(frame)
                video_writer.release()

            # a new form of success record
            for k in range(env_num):
                if i * env_num + k < self.n_eval:
                    num_success += int(dones[k])


        success_rate = num_success / eval_loop_num
        env.close()
        gc.collect()
        # print(f"[info] evaluate task {task_str} takes {t.get_elapsed_time():.1f} seconds")
        return success_rate

    def create_cfg_for_libero(self, task_embedding_format):
        self.cfg = DictConfig({'task_embedding_format': task_embedding_format,
                               'data': {'max_word_len': 25}})

        self.cfg.policy = OmegaConf.create()
        self.cfg.policy.language_encoder = OmegaConf.create()
        self.cfg.policy.language_encoder.network_kwargs = OmegaConf.create()


    def translate_obs_space(self, obs_space):

        translated_dict = {}
        translated_dict['rgb_obs'] = {}
        translated_dict['rgb_obs']['rgb_static'] = obs_space['agentview_image']
        translated_dict["rgb_obs"]['rgb_gripper'] = obs_space['robot0_eye_in_hand_image']
        translated_dict['robot_obs'] = obs_space['robot0_joint_pos']
        translated_dict['gripper_states'] = obs_space['robot0_gripper_qpos']
        translated_dict['depth_obs'] = {}

        return translated_dict

    def apply_transforms(self, data, train=False):
        # Assuming data contains images in 'rgb_static' and 'rgb_gripper'
        for key in data['rgb_obs']:
            # print(key)
            x = data['rgb_obs'][key]
            if len(x.shape) == 3:
                x = np.expand_dims(x, axis=0)
                # print(x.shape)
            x = torch.from_numpy(x).byte().permute(0, 3, 1, 2)
            for transform in self.transforms[key]:
                x = transform(x)
            data['rgb_obs'][key] = x.unsqueeze(0).to(self.device)
            # data['rgb_obs'][key] = transforms[key](data['rgb_obs'][key])

        return data

    def process_env_obs(self, env_obs, lang_embed, lang_text=None):
        return_obs = self.translate_obs_space(env_obs)
        return_obs = self.apply_transforms(return_obs)

        goal = {}
        goal['lang_text'] = lang_text
        goal['lang'] = lang_embed

        return return_obs, goal

if __name__ == "__main__":
    import omegaconf
    os.environ.CUDA_VISIBLE_DEVICES = '5'

    from mode.evaluation.utils import load_pl_module_from_checkpoint

    transform_cfg = omegaconf.OmegaConf.load("/home/yagmurlu/code/MoDE_Calvin/conf/datamodule/transforms/libero_transforms.yaml")
    transforms = transform_cfg['val']

    model = load_pl_module_from_checkpoint(
        "/home/yagmurlu/code/MoDE_Calvin/logs/runs/2024-08-03/13-07-00",
        0
    ).to('cuda:0')

    print("Rollout Libero")
    rollout = RolloutLibero(
        "libero_10",
        {},
        10,
        10,
        5,
        5,
        2,
        10,
        1,
        False,
        'clip',
        False,
        True,
        'cuda:0'
    )

    rollout.transforms = hydra.utils.instantiate(transforms)
    rollout.evaluate_policy(model, store_video="/home/yagmurlu/code/MoDE_Calvin/zzT-2")

    print('end')