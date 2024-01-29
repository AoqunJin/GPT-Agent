import numpy as np
import pickle

print(np.load("./atari_data/AirRaid/$store$_action_ckpt.1").shape)


with open("./data/assembly-v2/collect_61.bin", "rb") as f:
    mdp_data = pickle.load(f)
print(mdp_data.keys())
# print(mdp_data.keys())
print(mdp_data["obss"][0].shape)
print(mdp_data["acts"][0].shape)