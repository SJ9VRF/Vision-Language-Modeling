# early_fusion_model.py
import torch
import torch.nn as nn
import math

class EasyQAEarlyFusionNetwork(nn.Module):
    def __init__(self):
        super(EasyQAEarlyFusionNetwork, self).__init__()
        self.dropout = nn.Dropout(0.3)
        self.vision_projection = nn.Linear(2048, 768)
        self.text_projection = nn.Linear(512, 768)
        self.fc1 = nn.Linear(768, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.classifier = nn.Linear(256, 13)
        W = torch.Tensor(768, 768)
        self.W = nn.Parameter(W)
        self.relu_f = nn.ReLU()
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, image_emb, text_emb):
        Xv = self.relu_f(self.vision_projection(image_emb))
        Xt = self.relu_f(self.text_projection(text_emb))
        Xvt = Xv * Xt
        Xvt = self.relu_f(torch.mm(Xvt, self.W.t()))
        Xvt = self.fc1(Xvt)
        Xvt = self.bn1(Xvt)
        Xvt = self.dropout(Xvt)
        Xvt = self.classifier(Xvt)
        return Xvt

