import pickle
import copy
import random
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class MarkovDataset(Dataset):
    def __init__(self, folder_list, train_mode="base") -> None:
        super().__init__()
        if not isinstance(folder_list, list):
            raise TypeError
        if not train_mode in ["base", "sequence_modeling", "bilateral_modeling"]:
            raise ValueError
        self.train_mode = train_mode
        
        self.mdp_data_dirs = []
        for folder in folder_list:    
            file_list = (os.listdir(folder))
            for file in file_list:
                if file.endswith("bin"):
                    binary_data_dir = os.path.join(folder, file)
                    self.mdp_data_dirs.append(binary_data_dir)

    def __len__(self):
        return len(self.mdp_data_dirs)

    def __getitem__(self, index):
        with open(self.mdp_data_dirs[index], "rb") as f:
            mdp_data = pickle.load(f) 

        if self.train_mode == "base":
            data = mdp_data["obss"]
            position_id = [i+1 for i in range(len(data))]
            token_type_ids = [0 for i in range(len(data))]
            label = mdp_data["acts"]
        
        else:
            if (self.train_mode == "sequence_modeling") or \
                (self.train_mode == "bilateral_modeling" and random.choice([True, False])):
                seq = []
                for idx in range(len(mdp_data["obss"])):
                    seq.append(mdp_data["obss"][idx])
                    seq.append(mdp_data["acts"][idx])
                data = seq[:-1]
                position_id = [i+1 for i in range(len(mdp_data["obss"])) for _ in range(2)][:-1]
                label = seq[1:]
            else:
                data, label = [], []
                for idx in reversed(range(len(mdp_data["obss"])-1)):
                    data.append(mdp_data["obss"][idx])
                    data.append(mdp_data["acts"][idx])
                data.pop()
                position_id = [i+1 for i in reversed(range(len(mdp_data["obss"]) - 1)) for _ in range(2)][:-1]
                for idx in reversed(range(len(mdp_data["obss"])-1)):
                    label.append(mdp_data["acts"][idx])
                    label.append(mdp_data["obss"][idx+1])
                label.pop()

            token_type_ids = [1 if i % 2 == 0 else 2 for i in range(len(data))]

        return {
            "data": data, 
            "position_id": position_id,
            "attention_mask": [0 for i in range(len(data))],
            "token_type_ids": token_type_ids,
            "label": label
        }


def collate_batch(data):
    max_len = max(map(lambda x: len(x["data"]), data))
    bath_size = len(data)
    
    for bs in range(bath_size):
        if len(data[bs]["data"]) < max_len:
            padd_l = max_len - len(data[bs]["data"])
            # padding data
            data[bs]["data"].extend(np.zeros_like(data[bs]["data"][1]) if i % 2 == 0 
                                    else np.zeros_like(data[bs]["data"][0]) for i in range(padd_l))            
            # padding label
            data[bs]["label"].extend(np.zeros_like(data[bs]["label"][1]) if i % 2 == 0 
                                    else np.zeros_like(data[bs]["label"][0]) for i in range(padd_l))  
            # padding position_id
            data[bs]["position_id"].extend(0 for _ in range(padd_l))
            # padding attention_mask
            data[bs]["attention_mask"].extend(1 for _ in range(padd_l))
            # padding token_type_ids
            data[bs]["token_type_ids"].extend(0 for _ in range(padd_l))

    ret_dict = {
        "data": [np.array([data[bs]["data"][pid] for bs in range(bath_size)]) for pid in range(max_len)],
        "position_id": np.array([data[bs]["position_id"] for bs in range(bath_size)]),
        "attention_mask": np.array([data[bs]["attention_mask"] for bs in range(bath_size)]),
        "token_type_ids": np.array([data[bs]["token_type_ids"] for bs in range(bath_size)]),
        "label": [np.array([data[bs]["label"][pid] for bs in range(bath_size)]) for pid in range(max_len)],
    }

    return ret_dict


if __name__ == "__main__":
    
    # "base", "sequence_modeling", "bilateral_modeling"
    dataset = MarkovDataset(folder_list=["./data_collection/demos/assembly-v2"], train_mode="base")
    dataloader = DataLoader(dataset=dataset, batch_size=3, collate_fn=collate_batch, num_workers=0)
    for data in dataloader:
        ...
    print("Finish")