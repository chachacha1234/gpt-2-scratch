import math
import inspect
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module): # self attention layer 블럭
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # 입력 토큰의 차원을 768 -> 2304차원으로 늘리고 Q,K,V로 나눔
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # 각 head의 결과를 합침
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x) 
        q,k,v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # head 개수대로 q,k,v의 차원 나눔 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # 768 // 12 = 64
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        # pytorch의 함수로 attention 계산 // is_causal=True -> t 시점의 출력에서 t와 그 앞까지만 참조하도록.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        y = y.transpose(1,2).contiguous().view(B,T,C) # 각 head의 출력을 이어 붙임.
        y = self.c_proj(y) # 이어 붙인 output들을 선형 변환
        return y


class MLP(nn.Module): # sub_layer output -> 
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # 768 -> 3072
        self.gelu = nn.GELU(approximate='tanh') # GELU로 비선형성 추가
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # 3072 -> 768 원래 차원으로
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module): # Attention과 MLP를 하나의 블록으로 묶음
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # pre-LN으로 먼저 Layernorm 적용
        self.attn = CausalSelfAttention(config) # causal selfattention layer 통과
        self.ln_2 = nn.LayerNorm(config.n_embd)  
        self.mlp = MLP(config) # mlp layer 통과
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig: # 모델 크기 설정
    block_size: int = 1024 # 최대 입력 길이
    vocab_size: int = 50304 # 사용할 토큰 종류의 개수
    n_layers: int = 12 # transformer decoder의 블록 개수
    n_head: int = 12 # attention head의 개수
    n_embd: int = 768 # 각 토큰 별 표현 차원


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict( # GPT-2의 구조 뼈대 같은 것.
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embedding 번호에 해당하는 768차원의 벡터
            wpe = nn.Embedding(config.block_size, config.n_embd),   # positional embedding 번호에 해당하는 768차원의 벡터
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]), # Block 12개를 순서대로 보관
            ln_f = nn.LayerNorm(config.n_embd), # 마지막 블록의 output 정규화
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 다음 생성할 토큰들의 점수
        # input의 토큰과 output 후보의 토큰에 대한 점수를 매길 때 같은 가중치 사용을 위한 공유
        self.transformer.wte.weight = self.lm_head.weight 
        self.apply(self._init_weights) 
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # 학습 시작할 때 가중치 초기화
            std = 0.02 # 표준편차 0.02인 정규분포로 샘플링.
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    # 입력 토큰 idx를 받아 각 위치의 logits를 계산
    def forward(self, idx, targets=None):
        # 입력은 BxT 크기의 idx
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        # token embedding + position embedding을 입력으로 사용
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits = (B,T,vocab_size) 얻게되는 tensor
        loss = None
        if targets is not None: # 정답 토큰이 주어지면 loss도 계산
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    # 공개된 gpt-2의 가중치를 불러와서 모델에 복사
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        # 모델 타입에 따라 크기 설정
        config_args = {
            'gpt2': dict(n_layers=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layers=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layers=36, n_head=20, n_embd=1280),
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = cls(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # tensorflow를 pytorch로 변경하기 위해 하드코딩
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: HF={len(sd_keys_hf)}, mine={len(sd_keys)}"
        assert set(sd_keys_hf) == set(sd_keys), '가중치 이름이 일치하지 않습니다.'
        for k in sd_keys_hf: # hugging face의 gpt-2 가중치를 덮어씀
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        # gradient를 계산하는 학습 대상 parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Linear와 Embedding의 행렬 가중치엔 weight decay 적용
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # bias와 LayerNorm은 weight decay 적용 X
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [ # 그룹별 weight decay 적용 설정
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with{num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)


class DataLoaderLite: # 토큰 순서대로 배치 만들기
    def __init__(self, B, T, *, tokens=None, shards=None):
        self.B = B # 배치 당 시퀀스 개수
        self.T = T # 시퀀스 당 토큰 개수
        self.shards = shards # 파일 번호
        self.initial_tokens = tokens 
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        if self.shards is not None:
            self.tokens = load_tokens(self.shards[self.current_shard])
        else:
            self.tokens = self.initial_tokens

    def next_batch(self): # 마지막 input 토큰의 정답을 위해 B*T+1개를 읽음.
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T) # 입력과 정답이 한 칸 어긋남
        self.current_position += B*T # 마지막 정답 토큰이 다음 배치의 첫 입력

        # 다음 배치를 만들 수 없으면 다음 shard 또는 처음으로 이동
        if self.current_position + B*T + 1 > len(self.tokens):
            if self.shards is not None:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


def get_lr(step, schedule):
    warmup = schedule['warmup_steps'] # 학습 초반 warmup 구간에서는 가중치를 조금씩 높임.
    if warmup and step < warmup:
        return schedule['lr'] * (step + 1) / warmup
    # warmup 이후에는 얼마나 진행했는지 ratio를 구해 cosine 곡선으로 학습률 낮춤.
    ratio = min(1.0, max(0.0, (step - warmup) / (schedule['decay_steps'] - warmup)))
    return schedule['min_lr'] + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (schedule['lr'] - schedule['min_lr'])
