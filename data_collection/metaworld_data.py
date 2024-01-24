import os
import pickle
import shutil
from typing import Tuple, List, Type, Optional, Any
import multiprocessing
import argparse

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


def get_all_envs(n_envs: int = 50) -> Tuple[List[str], List[Type]]:
    """Get all Metaworld 50 environments.

    Returns:
        Tuple: A tuple containing a list of environment names and a list of environment classes.
    """
    envs = []
    env_names = []
    for env_name in _env_dict.MT50_V2:
        env_names.append(env_name)
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
    """Collect data for a single trial in the specified environment using the given policy.

    Args:
        env (Type): The environment instance to collect data from.
        policy (Type): The policy used for data collection.
        env_name (str): The name of the environment.
        trail (int): The trial number.
        use_rgb (bool, optional): Flag indicating whether to use RGB observations. Defaults to True.
        res (tuple, optional): Resolution of RGB observations. Defaults to (640, 480).
        camera_name (str, optional): Name of the camera view. One of ["corner", "topview", "behindGripper", "gripperPOV"].
                                     Defaults to "corner".
        num_step (int, optional): Number of steps in the trial. Defaults to 501.
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
    del data


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
        num_workers (int): Number of worker processes for parallel data collection. Defaults to 5.
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

                # TODO optimize
                pool.apply_async(
                    func=collect_trail,
                    args=(env, policy, env_name, trail, use_rgb, 
                          res, camera_name, num_step)
                )
        pool.close()
        pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect meta-world demonstration.")

    # parser.add_argument("--env_names", nargs="+", help="List of environment names.")
    # parser.add_argument("--envs", nargs="+", help="List of environment classes.")
    # parser.add_argument("--policy_names", nargs="+", help="List of policy names.")
    parser.add_argument("--n_envs", type=int, default=50, help="Number of collect environment.")
    parser.add_argument("--use_rgb", action="store_true", help="Flag indicating whether to use RGB observations.")
    parser.add_argument("--res", nargs=2, type=int, default=[224, 224], help="Resolution of RGB observations.")
    parser.add_argument("--camera_name", choices=["corner", "topview", "behindGripper", "gripperPOV"], default="corner", help="Name of the camera view.")
    parser.add_argument("--num_trail", type=int, default=2000, help="Number of trials for data collection.")
    parser.add_argument("--num_step", type=int, default=501, help="Number of steps per trial.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store collected data.")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    env_names, envs = get_all_envs(args.n_envs)
    policy_names = get_policy_names(env_names)

    print("+" * 30, "parser", "+" * 30)
    print("n_envs", args.n_envs)
    print("env_names", env_names)
    print("policy_names", policy_names)    
    print("use_rgb", args.use_rgb)
    print("res", tuple(args.res))
    print("camera_name", args.camera_name)
    print("num_trail", args.num_trail)
    print("num_step", args.num_step)
    print("num_trail", args.num_trail)
    print("num_workers", args.num_workers)
    print("data_dir", args.data_dir)
    print("+" * 30, "parser", "+" * 30)

    collect_demos(env_names, envs, policy_names, use_rgb=args.use_rgb, res=tuple(args.res),
                  camera_name=args.camera_name, num_trail=args.num_trail, num_step=args.num_step,
                  num_workers=args.num_workers)
    print("All finished!")