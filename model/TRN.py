import torch.nn as nn
import math
from base import BaseModel
from model.mobilenetV2 import MobileNet2
from model.TRNmodule import TRN_module
class TRN(BaseModel):
    
    def __init__(self, n_class):
        super(TRN, self).__init__()
        self.mobilenet_v2 = MobileNet2()
        self.new_fc = nn.Linear(2048, 256)
        self.TRN_module = TRN_module(256, num_frames=8, num_class=n_class)
    def forward(self, x):
        
        x = self.mobilenet_v2(x)
        x = self.new_fc(x)
        x = self.TRN_module(x)
        
        return x
    

