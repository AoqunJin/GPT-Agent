import os
import pickle
import shutil
from typing import Tuple, List, Type, Optional, Any
import multiprocessing
import argparse
import logging
import math

from tqdm import tqdm
import numpy as np
from metaworld.policies import *
import metaworld.envs.mujoco.env_dict as _env_dict


def get_policy_names(env_names: List[str]) -> List[str]:
    """Generate policy names based on environment names.

    Args:
        env_names (List[str]): List of environment names.

    Returns:
        List[str]: List of corresponding policy names.
    """
    policy_names = []
    for env_name in env_names:
        base = "Sawyer"
        res = env_name.split("-")
        for substr in res:
            base += substr.capitalize()
        policy_name = base + "Policy"
        if policy_name == "SawyerPegInsertSideV2Policy":
            policy_name = "SawyerPegInsertionSideV2Policy"
        policy_names.append(policy_name)
    
    return policy_names


def get_all_envs(env_names: List[str] = None, n_envs: int = 50) -> Tuple[List[str], List[Type]]:
    """Get Metaworld environments.

    Args:
        env_names (List[str]): List of environment names.
        n_envs (int): Use number of environments

    Returns:
        Tuple: A tuple containing a list of environment names and a list of environment classes.
    """
    envs = []
    env_names = ([] if env_names is None else env_names)
    # get env names
    if env_names == []:
        for env_name in _env_dict.MT50_V2:
            env_names.append(env_name)
    if n_envs > len(env_names):
        raise ValueError(f"n_envs={n_envs} should less then "
                         f"len(env_names)={len(env_names)} with env_names={env_names}")
    # get envs
    for env_name in env_names:
        envs.append(_env_dict.MT50_V2[env_name])

    return env_names[:n_envs], envs[:n_envs]


def file_maker(folder: Optional[str] = None):
    """Create a folder and initialize the data directory.

    Args:
        folder (Optional[str]): Name of the folder to be created within the data directory.
                                If None, the data directory is initialized.
    """
    # Make folder
    if folder is not None:
        path = os.path.join(args.data_dir, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    # Init data dir
    elif not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        

def data_saver(file: str, data: Any):
    """Save data to a binary file.

    Args:
        file (str): Name of the file (without extension) to save the data.
        data: Data to be saved.
    """
    with open(os.path.join(args.data_dir, f"{file}.bin"), "wb") as f:
        pickle.dump(data, f)


def collect_trail(env, policy, env_name: str, trails_sta: int, trails_end: int,
                  use_rgb: bool = True, res: tuple = (224, 224), 
                  camera_name: str = "corner", num_step: int = 501):
    """Collect data for a single trial in the specified environment using the given policy.

    Args:
        env (Type): The environment instance to collect data from.
        policy (Type): The policy used for data collection.
        env_name (str): The name of the environment.
        trails_sta (int): The trial number, start index.
        trails_end (int): The trial number, end index.
        use_rgb (bool, optional): Flag indicating whether to use RGB observations. Defaults to True.
        res (tuple, optional): Resolution of RGB observations. Defaults to (640, 480).
        camera_name (str, optional): Name of the camera view. One of ["corner", "topview", "behindGripper", "gripperPOV"].
                                     Defaults to "corner".
        num_step (int, optional): Number of steps in the trial. Defaults to 501.
    """
    for trail in range(trails_sta, trails_end):
        obss, acts, rews, dones = [], [], [], []
        obs = env.reset()
        obs_agent = obs

        if use_rgb:
            obs_agent = env.sim.render(*res, mode="offscreen", camera_name=camera_name)[:,:,::-1]

        for i in range(num_step):
            obss.append(obs_agent)
            act = policy.get_action(obs)
            obs, rew, done, info = env.step(act+0.1*np.random.randn(4,))
            obs_agent = obs

            if use_rgb:
                obs_agent = env.sim.render(*res, mode="offscreen", camera_name=camera_name)[:,:,::-1]

            acts.append(act)
            rews.append(rew)

            if info["success"] == True:
                dones.append(True)
            else:
                dones.append(done)

            if done:
                break
        
        data = {
            "obss": obss, "acts": acts, "rews": rews, "dones": dones
        }

        data_saver(str(env_name) + "/collect_" + str(trail), data)
        del data


def collect_demos(env_names: List[str], envs: List[Type], policy_names: List[str],
                  use_rgb: bool = True, res: tuple = (224, 224), camera_name: str = "corner",
                  num_trail: int = 2000, num_step: int = 501, num_workers: int = 1) -> None:
    """Collect demonstration data for the specified environments using the given policies.

    Args:
        env_names (List[str]): List of environment names.
        envs (List[Type]): List of environment classes.
        policy_names (List[str]): List of policy names.
        use_rgb (bool): Flag indicating whether to use RGB observations. Defaults to True.
        res (tuple): Resolution of RGB observations. Defaults to (224, 224).
        camera_name (str): Name of the camera view. One of ["corner", "topview", "behindGripper", "gripperPOV"].
                          Defaults to "corner".
        num_trail (int): Number of trials for data collection. Defaults to 2000.
        num_step (int): Number of steps per trial. Defaults to 501.
        num_workers (int): Number of worker processes for parallel data collection. Defaults to 1.
    """
    for i in range(len(env_names)):
        env_name = env_names[i]
        env = envs[i]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        
        policy_name = policy_names[i]
        policy = globals()[policy_name]()

        file_maker(str(env_name))
        
        # bar 
        # TODO item level update
        pbar = tqdm(total=num_workers)
        pbar.set_description(env_name)
        update = lambda *x: pbar.update()

        # parallel
        segment_length = math.ceil(num_trail / num_workers)
        trails_sta, trails_end = 0, segment_length        
        pool = multiprocessing.Pool(processes=num_workers)
        for worker in range(num_workers):
            if worker == num_workers - 1:
                trails_end = num_trail
            pool.apply_async(
                func=collect_trail,  # collect trails [trails_sta, trails_end)
                args=(env, policy, env_name, trails_sta, trails_end, use_rgb, 
                        res, camera_name, num_step,),
                callback=update
            )
            trails_sta += segment_length
            trails_end += segment_length
        pool.close()
        pool.join()
        logging.info(f"{env_name} finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect meta-world demonstration.")

    parser.add_argument("--env_names", nargs="+", help="List of environment names.")
    parser.add_argument("--n_envs", type=int, default=50, help="Number of collect environment.")
    parser.add_argument("--use_rgb", action="store_true", help="Flag indicating whether to use RGB observations.")
    parser.add_argument("--res", nargs=2, type=int, default=[224, 224], help="Resolution of RGB observations.")
    parser.add_argument("--camera_name", choices=["corner", "topview", "behindGripper", "gripperPOV"], default="corner", help="Name of the camera view.")
    parser.add_argument("--num_trail", type=int, default=2000, help="Number of trials for data collection.")
    parser.add_argument("--num_step", type=int, default=501, help="Number of steps per trial.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--data_dir", type=str, default="metaworld_data", help="Directory to store collected data.")
    args = parser.parse_args()

    env_names, envs = get_all_envs(args.env_names, args.n_envs)
    policy_names = get_policy_names(env_names)

    logging.basicConfig(level=logging.DEBUG,  # level DEBUG
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"n_envs {args.n_envs}")
    logging.info(f"env_names {env_names}")
    logging.info(f"policy_names {policy_names}")    
    logging.info(f"use_rgb {args.use_rgb}")
    logging.info(f"res {tuple(args.res)}")
    logging.info(f"camera_name {args.camera_name}")
    logging.info(f"num_trail {args.num_trail}")
    logging.info(f"num_step {args.num_step}")
    logging.info(f"num_trail {args.num_trail}")
    logging.info(f"num_workers {args.num_workers}")
    logging.info(f"data_dir {args.data_dir}")

    collect_demos(env_names, envs, policy_names, use_rgb=args.use_rgb, res=tuple(args.res),
                  camera_name=args.camera_name, num_trail=args.num_trail, num_step=args.num_step,
                  num_workers=args.num_workers)
    
    logging.info("All finished!")