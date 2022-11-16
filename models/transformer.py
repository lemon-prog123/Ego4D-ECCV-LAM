
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,temporal=False,cls=False,no_encoder=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before,temporal,cls)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.no_encoder=no_encoder
        self._reset_parameters()
        self.temporal=temporal
        self.d_model = d_model
        self.nhead = nhead
        self.cls=cls

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed,B=0,T=0,H=0,W=0):
        
        if self.temporal and self.cls:
            bs,c, h, w = B,self.d_model,H,W
            memory = self.encoder(src, src_key_padding_mask=None, pos=None,B=B,T=T,H=H,W=W)# b (hwt) m
            query_embed=query_embed.permute(1,0,2) #[Num,Batch,Channel]
            encoder_out=memory[:,0,:]
            memory=rearrange(memory[:,1:], 'b (h w t) m -> (h w) t b m',b=B,h=H,w=W,t=T)
            memory=torch.mean(memory,1,False)
            tgt = torch.zeros_like(query_embed)
            
            hs = self.decoder(tgt, memory, memory_key_padding_mask=None,
                                pos=None, query_pos=query_embed)
            return hs.transpose(1, 2), encoder_out
        else:
            if self.temporal:
                bs,c, h, w = B,self.d_model,H,W
                src = rearrange(src, '(b t) m h w -> (h w t) b m',b=B,h=H,w=W,t=T)
                pos_embed=rearrange(pos_embed, '(b t) m h w -> (h w t) b m',b=B,h=H,w=W,t=T)
                query_embed=query_embed.permute(1,0,2) #[Num,Batch,Channel]
                mask=mask.flatten(1)
                memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed,B=B,T=T,H=H,W=W)
                
                memory = rearrange(memory, '(h w t) b m -> (h w) t b m',b=B,h=H,w=W,t=T)
                #memory=torch.mean(memory,1,False)
                memory=memory[:,3,:]
                #encoder_out=memory
                pos_embed=rearrange(pos_embed, '(h w t) b m -> (h w) t b m',b=B,h=H,w=W,t=T)
                pos_embed=pos_embed[:,0,:]
                mask=mask[:B,:]
                
            else:
                # flatten NxCxHxW to HWxNxC
                
                # src->feature map
                # mask->None
                # pos_embed ->pos embedding
                # query->head pose query
                
                bs, c, h, w = src.shape
                src = src.flatten(2).permute(2, 0, 1)
                pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
                query_embed=query_embed.permute(1,0,2)
                
                #query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
                if mask!=None:
                    mask = mask.flatten(1)
                    
                # feature map -> self attention
                if self.no_encoder:
                    memory=src
                else:
                    memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
                #print('Memory')
                #print(memory[:,0,:].shape)
                #print(memory[:,0,:])
                
            # init tgt
            tgt = torch.zeros_like(query_embed)
            # memory-> output of the encoder
            # mask->None
            # pos_embed->pos embedding
            #query_pos->query_embed
            
            hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                pos=pos_embed, query_pos=query_embed)

            return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,B=0,T=0,H=0,W=0):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos,B=B,T=T,H=H,W=W)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        #print('Output')
        #print(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,temporal=False,cls=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        if temporal==True:
            self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.temporal_norm = nn.LayerNorm(d_model)
            self.temporal_dropout=nn.Dropout(dropout)
            self.temporal_fc=nn.Linear(d_model,d_model)
            
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.temporal=temporal
        self.cls=cls
        
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,B=0,T=0,H=0,W=0):
        if self.temporal and self.cls:
            ##Temporal
            xt=src[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m ->t (b h w) m',b=B,h=H,w=W,t=T)
            qt=kt=self.with_pos_embed(xt,pos)
            
            res_temporal = self.temporal_attn(qt,kt,value=self.temporal_norm(xt),attn_mask=src_mask,key_padding_mask=None)[0]
            res_temporal = rearrange(res_temporal, 't (b h w) m -> (h w t) b m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            
            srct=rearrange(src[:,1:,:], 'b (h w t) m ->(h w t) b m',b=B,h=H,w=W,t=T)
            xt=srct+self.temporal_dropout(res_temporal)
            
            ##Spatial
            init_cls_token = src[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(0)
            xs = xt
            xs = rearrange(xs, '(h w t) b m -> (h w) (b t) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs),0)
            qs=ks=self.with_pos_embed(xs,pos)
            
            res_spatial=self.self_attn(qs,ks,value=self.norm1(xs),attn_mask=src_mask,key_padding_mask=None)[0]
            cls_token = res_spatial[0,:,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True)
            res_spatial = res_spatial[1:,:,:]
            res_spatial = rearrange(res_spatial, '(h w) (b t) m ->b (h w t) m',b=B,h=H,w=W,t=T)
            res=res_spatial
            x=rearrange(xt,'(h w t) b m -> b (h w t) m',b=B,h=H,w=W,t=T)
            
            x = torch.cat((init_cls_token, x), 1) + self.dropout1(torch.cat((cls_token, res), 1))
            
            src2=self.norm2(x)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
            #src=rearrange(src,'b (h w t) m -> (h w t) b m',b=B,h=H,w=W,t=T)
            
        elif self.temporal:
            #temporal
            srct=rearrange(src, '(h w t) b m -> t (b h w) m',b=B,h=H,w=W,t=T)
            pos=rearrange(pos, '(h w t) b m -> t (b h w) m',b=B,h=H,w=W,t=T)
            
            srct=self.temporal_norm(srct)
            qt = kt = self.with_pos_embed(srct, pos)
            res_temporal=self.temporal_attn(qt,kt,value=srct,attn_mask=src_mask,key_padding_mask=None)[0]
            res_temporal = rearrange(res_temporal, 't (b h w) m -> (h w t) b m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            srct=src+self.temporal_dropout(res_temporal)
            
            #spatial
            srcs=srct
            srcs = rearrange(srcs, '(h w t) b m -> (h w) (b t) m',b=B,h=H,w=W,t=T)
            pos = rearrange(pos, 't (b h w) m -> (h w) (b t) m',b=B,h=H,w=W,t=T)
            
            srcs=self.norm1(srcs)
            qs=ks=self.with_pos_embed(srcs,pos)
            res_spatial=self.self_attn(qs,ks,value=srcs,attn_mask=src_mask,key_padding_mask=None)[0]
            res_spatial = rearrange(res_spatial, '(h w) (b t) m -> (h w t) b m',b=B,h=H,w=W,t=T)
            
            res = res_spatial
            src=srct+self.dropout1(res)
            
            #src = rearrange(src, '(h w t) b m -> (h w) t b m',b=B,h=H,w=W,t=T)
            #src=torch.mean(src,1,False)# avgpool
            
            
            src2=self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
            
        else:
            src2 = self.norm1(src)
            q = k = self.with_pos_embed(src2, pos)
            src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,B=0,T=0,H=0,W=0):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos,B=B,T=T,H=H,W=W)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        #self attention
        #tgt 
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        
        #cross attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")