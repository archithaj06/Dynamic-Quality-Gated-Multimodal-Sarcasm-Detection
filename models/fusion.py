import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TextEncoder, VideoEncoder, AudioEncoder

class QualityGate(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

class DQGMSD(nn.Module):
    def __init__(self, proj_dim=256, gate_temp=0.75):
        super().__init__()
        self.text_enc = TextEncoder(proj_dim=proj_dim)
        self.vid_enc = VideoEncoder(proj_dim=proj_dim)
        self.aud_enc = AudioEncoder(proj_dim=proj_dim)
        self.gate_t, self.gate_v, self.gate_a = QualityGate(proj_dim), QualityGate(proj_dim), QualityGate(proj_dim)
        self.temp = gate_temp
        self.classifier = nn.Sequential(nn.Linear(proj_dim * 3, 512), nn.GELU(), nn.Linear(512, 2))

    def forward(self, ids, mask, video, audio):
        t_f, v_f, a_f = self.text_enc(ids, mask), self.vid_enc(video), self.aud_enc(audio)
        q = torch.cat([self.gate_t(t_f), self.gate_v(v_f), self.gate_a(a_f)], dim=1)
        weights = F.softmax(q / self.temp, dim=1)
        fused = torch.cat([t_f * weights[:, 0:1], v_f * weights[:, 1:2], a_f * weights[:, 2:3]], dim=1)
        return self.classifier(fused), weights