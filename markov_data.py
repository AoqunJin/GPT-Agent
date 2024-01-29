import pickle
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MarkovDataset(Dataset):
    def __init__(self, folder_list, train_mode="base", seq_len=128) -> None:
        """_summary_

        Args:
            folder_list (_type_): _description_
            train_mode (str, optional): _description_. Defaults to "base".

        The data file must end with `bin`, the step number > 1, the number of 
        elements in fields like `obss` `acts` must same.
        """
        super().__init__()
        if not isinstance(folder_list, list):
            raise TypeError
        if not train_mode in ["base", "sequence_modeling", "bilateral_modeling"]:
            raise ValueError
        self.train_mode = train_mode
        self.seq_len = seq_len
        
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

        # for idx in range(len(mdp_data["obss"])):
        #     mdp_data["obss"][idx] = mdp_data["obss"][idx].cpu()
        for idx in range(len(mdp_data["acts"])):
            mdp_data["acts"][idx] = mdp_data["acts"][idx].unsqueeze(0)

        # random clip
        if len(mdp_data["obss"]) > self.seq_len:
            random_clip_idx = random.randint(0, len(mdp_data["obss"]) - self.seq_len)
            mdp_data["obss"] = mdp_data["obss"][random_clip_idx: random_clip_idx + self.seq_len]
            mdp_data["acts"] = mdp_data["acts"][random_clip_idx: random_clip_idx + self.seq_len]

        if self.train_mode == "base":
            # data:  s_1, s_2, ..., s_n
            # label: a_1, a_2, ..., a_n
            data = mdp_data["obss"]
            position_id = [i+1 for i in range(len(data))]
            token_type_ids = [1 for i in range(len(data))]
            label = mdp_data["acts"]
        
        else:
            if (self.train_mode == "sequence_modeling") or \
                (self.train_mode == "bilateral_modeling" and random.choice([True, False])):
                # data:  s_1, a_1, s_2, a_2, ..., s_n
                # label:      a_1, s_2, a_2, ..., s_n, a_n
                seq = []
                for idx in range(len(mdp_data["obss"])):
                    seq.append(mdp_data["obss"][idx])
                    seq.append(mdp_data["acts"][idx])
                data = seq[:-1]
                position_id = [i+1 for i in range(len(mdp_data["obss"])) for _ in range(2)][:-1]
                label = seq[1:]
            else:
                # data:  s_{n-1}, a_{n-1}, ..., s_2, a_2, s_1
                # label: a_{n-1}, s_n, ...,     a_2, s_3, a_1
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
            data[bs]["data"].extend(torch.zeros_like(data[bs]["data"][1]) if i % 2 == 0 
                                    else torch.zeros_like(data[bs]["data"][0]) for i in range(padd_l))            
            # padding label
            data[bs]["label"].extend(torch.zeros_like(data[bs]["label"][1]) if i % 2 == 0 
                                    else torch.zeros_like(data[bs]["label"][0]) for i in range(padd_l))  
            # padding position_id
            data[bs]["position_id"].extend(0 for _ in range(padd_l))
            # padding attention_mask
            data[bs]["attention_mask"].extend(1 for _ in range(padd_l))
            # padding token_type_ids
            data[bs]["token_type_ids"].extend(0 for _ in range(padd_l))

    # print(data[bs]["label"])
    ret_dict = {
        "data": [torch.cat([data[bs]["data"][pid] for bs in range(bath_size)], dim=0) for pid in range(max_len)],
        "position_id": torch.LongTensor([data[bs]["position_id"] for bs in range(bath_size)]),
        "attention_mask": torch.LongTensor([data[bs]["attention_mask"] for bs in range(bath_size)]),
        "token_type_ids": torch.LongTensor([data[bs]["token_type_ids"] for bs in range(bath_size)]),
        "label": [torch.cat([data[bs]["label"][pid] for bs in range(bath_size)], dim=0) for pid in range(max_len)],
    }

    return ret_dict


if __name__ == "__main__":
    # "base", "sequence_modeling", "bilateral_modeling"
    dataset = MarkovDataset(folder_list=["./data_collection/data/assembly-v2"], train_mode="bilateral_modeling")
    dataloader = DataLoader(dataset=dataset, batch_size=3, collate_fn=collate_batch, num_workers=0)
    for data in dataloader:
        print(data["data"][0].size())
        print(data["label"][0].size())        
        break

    print("Finish")