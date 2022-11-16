import torch, os, math, logging,sys
import torch.nn.functional as F
from models.transformer import Transformer
from models.position_encoding import PositionEmbeddingSine
from models.misc import NestedTensor
import torch.nn as nn
import torch.optim
import torch.utils.data
from model.resnet import resnet18
from copy import deepcopy

logger = logging.getLogger(__name__)

from einops import rearrange
        
        
class GazeDETR(nn.Module):
    def __init__(self, args):
        super(GazeDETR, self).__init__()
        self.args = deepcopy(args)
        self.img_feature_dim = 256
        
        self.base_model = resnet18(pretrained=True,head_query=args.head_query,mask=args.mask)
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)
        
        if self.args.GRU:
            self.lstm=nn.GRU(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
        else:
            self.lstm = nn.LSTM(self.img_feature_dim, self.img_feature_dim,bidirectional=True,num_layers=2,batch_first=True)
            
        self.running_mean=torch.zeros(1)
        self.running_var=torch.ones(1)
        self.yaw_embed=nn.Embedding(91,2*self.img_feature_dim)
        self.pitch_embed=nn.Embedding(91,2*self.img_feature_dim)
        self.roll_embed=nn.Embedding(91,2*self.img_feature_dim)
        self.temporal=args.temporal
        self.transformer=Transformer(
                            d_model=512,
                            dropout=0.1,
                            nhead=8,
                            dim_feedforward=args.dim_feedforward,
                            num_encoder_layers=args.num_encoder_layers,
                            num_decoder_layers=args.num_decoder_layers,
                            normalize_before=True,
                            return_intermediate_dec=False,temporal=self.temporal,cls=args.cls)
        
        
        self.input_proj = nn.Conv2d(512,512, kernel_size=1)
        
        hidden_dim=2*self.img_feature_dim
        N_steps = hidden_dim // 2
        self.position_embedding=PositionEmbeddingSine(N_steps, normalize=True)

        self.last_layer3=nn.Linear(6*self.img_feature_dim, 2*self.img_feature_dim)# 3*512 - 512

        self.last_layer1=nn.Linear(4* self.img_feature_dim, 128) # 1024 -128
        self.last_layer2 = nn.Linear(128, 2) # 128

        self.load_checkpoint()
        
    
    def forward(self, input,query=None,hidden=None):
        
        yaw_query=self.yaw_embed(query[:,0]).unsqueeze(1)
        pitch_query=self.pitch_embed(query[:,1]).unsqueeze(1)
        roll_query=self.roll_embed(query[:,2]).unsqueeze(1)
        query=torch.cat((yaw_query,pitch_query,roll_query),1) # [batch_size , 3 , 512]
        
        base_out,x=self.base_model(input.view((-1, 3) + input.size()[-2:]))
        
        x=x.view(input.size(0),7,512,7,7)[:,3]
        b, c, h, w =x.shape
        mask = torch.zeros((b,224,224), dtype=torch.bool,device=x.device)
        mask = F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x= NestedTensor(x, mask)
        pos =(self.position_embedding(x).to(x.tensors.dtype)).cuda()
        src, mask = x.decompose()
        hs = self.transformer(self.input_proj(src), mask,query, pos)[0].squeeze()

        if len(hs.shape)==2:
            hs=hs.unsqueeze(0)
            
        hs=hs.flatten(1)
        head_out=self.last_layer3(hs)

        base_out = base_out.view(input.size(0),7,self.img_feature_dim)
            
        lstm_out, _ = self.lstm(base_out)
        lstm_out = lstm_out[:,3,:]
        lstm_out=lstm_out.to(torch.float32)
            
        lstm_out=torch.cat((lstm_out,head_out),1)
        output = self.last_layer1(lstm_out)
            
        output = self.last_layer2(output).view(-1,2)
             
        return output
    
    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f'loading checkpoint {self.args.checkpoint}')
                map_loc = f'cuda:{self.args.rank}' if torch.cuda.is_available() else 'cpu'
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if 'Pure' in self.args.checkpoint:
                    state_dict=state
                else:
                    if 'module' in list(state["state_dict"].keys())[0]:
                        state_dict = { k[7:]: v for k, v in state["state_dict"].items() }
                    else:
                        state_dict = state["state_dict"]
                        
                names=[]
                if (('gaze360_model.pth' in self.args.checkpoint) or ('home' in self.args.checkpoint)) and not self.args.gaze and not self.args.val and not self.args.eval:
                    state_dict.pop('last_layer.weight')
                    state_dict.pop('last_layer.bias')
                    if self.args.GRU and ('home' not in self.args.checkpoint):
                        for name,value in state_dict.items():
                            if 'lstm' in name:
                                names.append(name)
                        for name in names:
                            state_dict.pop(name)
                print(state_dict.keys())
                self.load_state_dict(state_dict, strict=False)    
                


