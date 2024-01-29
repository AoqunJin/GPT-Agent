from typing import Iterable, Any
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


from markov_data import MarkovDataset, DataLoader, collate_batch
from decoder_only import DecoderOnly, DecoderOnlyLayer


class MDPEmbedding(nn.Module):
    def __init__(self, train_mode) -> None:
        super().__init__()
        self.train_mode = train_mode

        # State embedding
        self.state_embedding = nn.Linear(state_space_size, hidden_size)

        # Action embedding
        if self.train_mode != "base":
            self.action_embedding = nn.Sequential(
                nn.Linear(action_space_size, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, hidden_size)
            )

        # Position embedding
        self.position_embedding = nn.Embedding(
            num_embeddings=max_seq_len + 1,
            embedding_dim=hidden_size,
            padding_idx=0
        )

        # Type embedding
        self.type_embedding = nn.Embedding(
            num_embeddings=max_seq_len + 1,
            embedding_dim=hidden_size,
            padding_idx=0
        )

    def forward(self, data, position_id, token_type_ids, **argv):
        ret_list = []
        for idx, item in enumerate(data):
            if idx % 2 == 0 or self.train_mode == "base":
                item = self.state_embedding(item.float())
            else:
                item = self.action_embedding(item)
            ret_list.append(item.unsqueeze(1))

        position_embedding = self.position_embedding(position_id)
        token_embedding = self.type_embedding(token_type_ids)

        return torch.cat(ret_list, dim=1) + position_embedding + token_embedding


class MultiValHead(nn.Module):
    def __init__(self, train_mode) -> None:
        super().__init__()
        self.train_mode = train_mode
        if self.train_mode != "base":
            self.state_ffn = nn.Sequential(
                nn.Linear(hidden_size, 2048),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(2048, state_space_size)
            )
        self.action_ffn = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, action_space_size)
        )
    
    def forward(self, x):
        if self.train_mode == "base":
            return self.action_ffn(x)
        else:
            return self.action_ffn(x[:, ::2]), self.state_ffn(x[:, 1::2])


def MDP_predictionLoss(x, label, mode, mse_weight=1) -> torch.Tensor:
    """ The Markov decision process loss """
    for idx in range(len(label)):
        label[idx] = label[idx].unsqueeze(1)
    if mode == "base":
        label = torch.cat(label, dim=1)
        loss = F.mse_loss(x, label)
    else:
        label1 = torch.cat(label[::2], dim=1)
        label2 = torch.cat(label[1::2], dim=1)
        loss = F.mse_loss(x[0], label1) + F.mse_loss(x[1], label2)

    return loss * mse_weight


class GPTModel(nn.Module):
    def __init__(self, train_mode, num_layers, hidden_size, num_attention_heads) -> None:
        super().__init__()
        if not train_mode in ["base", "sequence_modeling", "bilateral_modeling"]:
            raise ValueError

        self.embedding = MDPEmbedding(train_mode=train_mode)

        self.decoder = DecoderOnly(
            num_layers=num_layers,
            d_model=hidden_size,
            nhead=num_attention_heads,
            batch_first=True
        )

    def forward(self, x):
        embedding = ...

    def act():
        ...

if __name__ == "__main__":
    bs = 2
    num_layers = 6
    hidden_size = 768
    num_attention_heads = 12
    action_space_size = 4
    state_space_size = 512
    max_seq_len = 128  # 128 256 512
    train_mode = "bilateral_modeling"
    # model = GPTModel(
    #     num_layers=num_layers,
    #     hidden_size=hidden_size,
    #     num_attention_heads=num_attention_heads
    # )
    with torch.no_grad():
        embedd_model = MDPEmbedding(train_mode=train_mode).to("cuda")
        head_model = MultiValHead(train_mode=train_mode).to("cuda")
        # "base", "sequence_modeling", "bilateral_modeling"
        dataset = MarkovDataset(
            folder_list=[
                "./data_collection/data/assembly-v2",
            ], 
            train_mode=train_mode,
            seq_len=max_seq_len
        )
        dataloader = DataLoader(dataset=dataset, batch_size=bs, collate_fn=collate_batch, pin_memory=True)
        for data in dataloader:

            data["data"] = [i.to("cuda", non_blocking=True) for i in data["data"]]
            data["label"] = [i.to("cuda", non_blocking=True) for i in data["label"]]
            data["position_id"] = data["position_id"].to("cuda", non_blocking=True)
            data["attention_mask"] = data["position_id"].to("cuda", non_blocking=True)
            data["token_type_ids"] = data["position_id"].to("cuda", non_blocking=True)

            print(data["data"][1].size())
            print(data["label"][1].size())
            # break

            sta = time.time()
            out = embedd_model(**data)
            print(out.size(), time.time() - sta, "s")

            sta = time.time()
            out = head_model(out)
            # print(out.size(), time.time() - sta, "s")
            print(out[0].size(), out[1].size(), time.time() - sta, "s")

            sta = time.time()
            loss = MDP_predictionLoss(out, data["label"], train_mode)
            print(loss, time.time() - sta, "s")

            break
        print("Finish!")