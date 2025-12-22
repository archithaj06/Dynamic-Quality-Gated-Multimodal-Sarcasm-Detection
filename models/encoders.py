import torch
import torch.nn as nn
from transformers import BertModel, Wav2Vec2Model
import torchvision.models as tv_models

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", proj_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.bert.config.hidden_size, proj_dim)

    def forward(self, ids, mask):
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.proj(out.last_hidden_state[:, 0])

class VideoEncoder(nn.Module):
    def __init__(self, proj_dim=256):
        super().__init__()
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(2048, proj_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x).flatten(1)
        return self.proj(feats.view(B, T, -1).mean(dim=1))

class AudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h", proj_dim=256):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.proj = nn.Linear(self.wav2vec.config.hidden_size, proj_dim)

    def forward(self, x):
        mask = (x != 0).long()
        out = self.wav2vec(x, attention_mask=mask).last_hidden_state
        return self.proj(out.mean(dim=1))