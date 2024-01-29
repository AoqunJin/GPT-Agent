import os
import pickle
import math

import numpy as np
import torch
from torch import multiprocessing
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import clip


def embedding(clip_model, traspose, device, folder, file_list):
    with torch.no_grad():
        for file in file_list:
            if file.endswith("bin"):
                binary_data_dir = os.path.join(folder, file)
                with open(binary_data_dir, "rb+") as f:
                    mdp_data = pickle.load(f)
                    obss = mdp_data["obss"]
                    acts = mdp_data["acts"]

                    # encode image
                    for i in range(len(obss)):
                        item = obss[i]
                        # TODO unsqueeze(0) ▶ batch
                        norm_item = traspose(Image.fromarray(item.transpose(2, 0, 1), mode="RGB")).unsqueeze(0).to(device)
                        item = clip_model.encode_image(norm_item)
                        obss[i] = item.cpu()  # TODO .squeeze(0)
                        acts[i] = torch.Tensor(acts[i])
                    mdp_data["obss"] = obss
                    mdp_data["acts"] = acts

                    # f.seek(0)
                    pickle.dump(mdp_data, f)
                    print(binary_data_dir)
                    


def process(folder_list, device, num_workers) -> None:
    multiprocessing.set_start_method("spawn")

    clip_model, traspose = clip.load("ViT-B/32", device=device)

    if not isinstance(folder_list, list):
        raise TypeError

    for folder in folder_list:    
        file_list = os.listdir(folder)

        segment_length = math.ceil((len(file_list)) / num_workers)
        trails_sta, trails_end = 0, segment_length        
        # pool = multiprocessing.Pool(processes=num_workers)
        for worker in range(num_workers):
            if worker == num_workers - 1:
                trails_end = len(file_list)

            # pool.apply_async(
            #     func=embedding,
            #     args=(clip_model, traspose, device, 
            #           folder, file_list[trails_sta: trails_end],),
            #     # callback=update
            # )
            embedding(clip_model, traspose, device, 
                      folder, file_list[trails_sta: trails_end])
            trails_sta += segment_length
            trails_end += segment_length
        # pool.close()
        # pool.join()

        print(f"{folder} Finished!")


if __name__ == "__main__":
    process(
        folder_list=[
            "./data_collection/data/assembly-v2",
            "./data_collection/data/basketball-v2",
            "./data_collection/data/bin-picking-v2"
        ],
        device="cuda",
        num_workers=1
    )
