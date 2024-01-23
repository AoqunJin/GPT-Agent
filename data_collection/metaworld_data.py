import os
import pickle
import shutil
from typing import Tuple, List, Type, Optional, Any
import multiprocessing
import argparse

import numpy as np
from metaworld.policies import *
import metaworld.envs.mujoco.env_dict as _env_dict


# TODO arg dir
DATA_DIR = "./demos"
# TODO parallel


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


def get_all_envs() -> Tuple[List[str], List[Type]]:
    """Get all Metaworld 50 environments.

    Returns:
        Tuple: A tuple containing a list of environment names and a list of environment classes.
    """
    envs = []
    env_names = []
    for env_name in _env_dict.MT50_V2:
        env_names.append(env_name)
        envs.append(_env_dict.MT50_V2[env_name])
    return env_names, envs


def file_maker(folder: Optional[str] = None):
    """Create a folder and initialize the data directory.

    Args:
        folder (Optional[str]): Name of the folder to be created within the data directory.
                                If None, the data directory is initialized.
    """
    # Make folder
    if folder is not None:
        path = os.path.join(DATA_DIR, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    # Init data dir
    elif not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        

def data_saver(file: str, data: Any):
    """Save data to a binary file.

    Args:
        file (str): Name of the file (without extension) to save the data.
        data: Data to be saved.
    """
    with open(os.path.join(DATA_DIR, f"{file}.bin"), "wb") as f:
        pickle.dump(data, f)


def collect_trail(env, policy, env_name: str, trail: int, 
                  use_rgb: bool = True, res: tuple = (640, 480), 
                  camera_name: str = "corner", num_step: int = 501):
    """_summary_

    Args:
        env (_type_): _description_
        policy (_type_): _description_
        env_name (str): _description_
        trail (int): _description_
        use_rgb (bool, optional): _description_. Defaults to True.
        res (tuple, optional): _description_. Defaults to (640, 480).
        camera_name (str, optional): _description_. Defaults to "corner".
        num_step (int, optional): _description_. Defaults to 501.
    """
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
        "obss": obss,
        "acts": acts,
        "rews": rews,
        "dones": dones
    }

    data_saver(str(env_name) + "/collect_" + str(trail), data)


def collect_demos(env_names: List[str], envs: List[Type], policy_names: List[str],
                  use_rgb: bool = True, res: tuple = (640, 480), camera_name: str = "corner",
                  num_trail: int = 2000, num_step: int = 501, num_workers: int = 5) -> None:
    """Collect demonstration data for the specified environments using the given policies.

    Args:
        env_names (List[str]): List of environment names.
        envs (List[Type]): List of environment classes.
        policy_names (List[str]): List of policy names.
        use_rgb (bool): Flag indicating whether to use RGB observations. Defaults to True.
        res (tuple): Resolution of RGB observations. Defaults to (640, 480).
        camera_name (str): Name of the camera view. One of ["corner", "topview", "behindGripper", "gripperPOV"].
                          Defaults to "corner".
        num_trail (int): Number of trials for data collection. Defaults to 2000.
        num_step (int): Number of steps per trial. Defaults to 501.
        num_workers
    """
    with multiprocessing.Pool(processes=num_workers) as pool:
        for i in range(len(env_names)):
            env_name = env_names[i]
            env = envs[i]()
            env._partially_observable = False
            env._freeze_rand_vec = False
            env._set_task_called = True
            
            policy_name = policy_names[i]
            policy = globals()[policy_name]()

            file_maker(str(env_name))

            for trail in range(num_trail):
                pool.apply_async(
                    func=collect_trail,
                    args=(env, policy, env_name, trail, use_rgb, 
                          res, camera_name, num_step)
                )
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect meta-world demonstration.")

    parser.add_argument("--num_trail", type=int, default=20, help="Number of trials for data collection.")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of worker processes.")
    args = parser.parse_args()

    env_names, envs = get_all_envs()
    policy_names = get_policy_names(env_names)
    collect_demos(env_names, envs, policy_names, num_trail=args.num_trail, num_workers=args.num_workers)