from typing import Iterable, Any
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from markov_data import MarkovDataset, DataLoader, collate_batch
from decoder_only import DecoderOnly, DecoderOnlyLayer


class MDPPredictionLoss():
    """ The Markov decision process loss """
    def __call__(self, mdp_predict_high_dim: torch.Tensor,
                 mdp: Iterable[torch.Tensor], 
                 enropy_weight=1, mse_weight=1) -> torch.Tensor:
        
        print(mdp_predict_high_dim.size())

        print(mdp_predict_high_dim[:, ::2].size())
        print(mdp_predict_high_dim[:, 1::2].size())

        print(torch.cat(mdp[1::2], dim=1).size())
        print(torch.cat(mdp[::2], dim=1).size())

        action_enropy = nn.CrossEntropyLoss(
            input=mdp_predict_high_dim[:-1:2],
            target=torch.Tensor(mdp[1::2])
        )
        # state_mse = nn.MSELoss(

        # )
        
        if enropy_weight <= 0 or mse_weight <= 0:
            raise ValueError

        return action_enropy * enropy_weight # + state_mse * mse_weight



class MDPEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(
            num_embeddings=action_space_size + 1,
            embedding_dim=hidden_size,
            padding_idx=action_space_size
        )
        # self.position_embedding = nn.Embedding(
        #     num_embeddings=action_space_size + 1,
        #     embedding_dim=hidden_size,
        #     padding_idx=action_space_size
        # )
        # self.type_embedding = nn.Embedding(
        #     num_embeddings=action_space_size + 1,
        #     embedding_dim=hidden_size,
        #     padding_idx=action_space_size
        # )
    def forward(self, input):
        # Can Only Embedding sparse value action to State dim
        # print(input[1::2])
        input = copy.deepcopy(input)
        
        embedding = []
        if "acts" in use_fields:
            if hasattr(self, "action_embedding"):
                ...

        for idx in range(len(input)):
            if idx % 2 != 0:
                input[idx] = self.action_embedding(input[idx])
            else:
                input[idx] = input[idx].unsqueeze(1)
        # for i in input:
        #     print(i.size())

        return torch.cat(input, dim=1)


class GPTBackbone(nn.Module):
    def __init__(self, num_layers, hidden_size, num_attention_heads) -> None:
        super().__init__()
        # Preprocess Markov decision process to tenssor
        self.embedding = MDPEmbedding()
        # Modeling
        self.decoder_layer = DecoderOnlyLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            batch_first=True
        )
        self.decoder = DecoderOnly(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size, eps=1e-5)
        )
        self.state_ffn = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, hidden_size)
        )
        self.action_ffn = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, hidden_size)
        )
        # Loss fn
        self.loss_fn = MDPPredictionLoss

    def forward(self, data, position_id, attention_mask, token_type_ids, label=True):
        # state action pair $(s_i, a_i)$ to tensor $(bs, mdp_len, hidden_size)$
        mdp_high_dim = self.embedding(mdp)
        print(mdp_high_dim.size())
        # exit()
        mdp_mask = torch.ones(size=(bs, mdp_high_dim.size(1)))
        mdp_predict_high_dim = self.decoder(mdp_high_dim, mdp_mask)

        if auto_loss_compute:
            self.loss_fn(mdp, mdp_predict_high_dim)

        return mdp_predict_high_dim

class MultiValHead(nn.Module):
    ...

class GPTModel(nn.Module):
    def __init__(self, train_mode, **argv) -> None:
        super().__init__()
        if not train_mode in ["base", "sequence_modeling", "bilateral_modeling"]:
            raise ValueError
        


if __name__ == "__main__":
    ...
    bs = 4
    num_layers = 12
    hidden_size = 768
    num_attention_heads = 12
    action_space_size = 42

    model = GPT(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads
    )
    use_fields = ["obss", "acts"]
    dataset = MarkovDataset(folder_list=["./dataset_1", "./dataset_2"], fields=use_fields)
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_batch)

    # loss_fn = MDPPredictionLoss()
    for data in dataloader:
        out = model(data, auto_loss_compute=False)
    # loss_fn(out, input)
