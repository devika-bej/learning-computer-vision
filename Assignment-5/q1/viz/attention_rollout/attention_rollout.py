import torch
import math
import numpy as np
from torchvision import transforms
import os

import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# ========= Set device =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("Using device:", device)

# ========= Define Modules =========
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dim_key = self.model_dim // self.num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(model_dim, model_dim)
        self.key = nn.Linear(model_dim, model_dim)
        self.value = nn.Linear(model_dim, model_dim)
        self.out = nn.Linear(model_dim, model_dim)
        self.attn_weights = None

    def attention_val(self, Q, K, V):
        score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_key)
        attn_weight = self.dropout(torch.softmax(score, dim=-1))
        new_val = torch.matmul(attn_weight, V)
        return new_val, attn_weight

    def split_layers(self, x):
        batch_size, seq_len, model_dim = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.dim_key).transpose(1, 2)

    def combine_layers(self, x):
        batch_size, _, seq_len, dim_key = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

    def forward(self, Q, K, V):
        Q = self.split_layers(self.query(Q))
        K = self.split_layers(self.key(K))
        V = self.split_layers(self.value(V))
        layer_out, self.attn_weights = self.attention_val(Q, K, V)
        final_output = self.out(self.combine_layers(layer_out))
        return final_output


class FeedForward(nn.Module):
    def __init__(self, model_dim, hid_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(model_dim, hid_dim)
        self.ac1 = nn.GELU()
        self.l2 = nn.Linear(hid_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        inp = self.l1(inp)
        inp = self.dropout(self.ac1(inp))
        inp = self.l2(inp)
        inp = self.dropout(inp)
        return inp


class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, hid_dim, dropout):
        super(Encoder, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        inp_norm = self.norm(inp)
        att_score = self.self_attn(inp_norm, inp_norm, inp_norm)
        inp = inp + self.dropout(att_score)
        inp_norm = self.norm(inp)
        ffn_out = self.ffn(inp_norm)
        inp = inp + self.dropout(ffn_out)
        return inp, self.self_attn.attn_weights


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, model_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, model_dim)

    def forward(self, x):
        B, C, H, W = x.size()
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch size ({self.patch_size}).")
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B, self.num_patches, self.patch_dim)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, model_dim, num_layer, num_heads, hid_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, model_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_embed = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_patches + 1, model_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoders = nn.ModuleList([Encoder(model_dim, num_heads, hid_dim, dropout) for _ in range(num_layer)])

    def forward(self, src):
        patch_emb = self.patch_embed(src)
        B = patch_emb.size(0)
        cls_tokens = self.cls_embed.expand(B, -1, -1)
        src_emb = torch.cat((cls_tokens, patch_emb), dim=1)
        src_emb = self.dropout(src_emb + self.pos_enc)
        attn_weights = []
        for encoder in self.encoders:
            src_emb, attn = encoder(src_emb)
            attn_weights.append(attn)
        return src_emb[:, 0], attn_weights


class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, num_classes, model_dim=768, num_layer=12, num_heads=12, hid_dim=3072, dropout=0.1):
        super(ViT, self).__init__()
        self.transformer = Transformer(img_size, patch_size, in_channels, model_dim, num_layer, num_heads, hid_dim, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x, attn_weights = self.transformer(x)
        x = self.norm(x)
        x = self.fc(x)
        return x, attn_weights


# ========= Load Model =========
model = ViT(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    model_dim=192,
    num_layer=9,
    num_heads=12,
    hid_dim=768,
    dropout=0.1
)
model_path = "80percent_above.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ========= Load a sample image from CIFAR-10 test set =========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
cifar10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
image, label = cifar10[0]
input_tensor = image.unsqueeze(0).to(device)

# ========= Attention Rollout =========
def compute_attention_rollout(attn_weights):
    rollout = torch.eye(attn_weights[0].size(-1)).to(device)
    for attn in attn_weights:
        attn_heads_mean = attn.mean(dim=1)  # Average over heads
        attn_heads_mean += torch.eye(attn_heads_mean.size(-1)).to(device)  # Add identity for residual connection
        attn_heads_mean /= attn_heads_mean.sum(dim=-1, keepdim=True)  # Normalize
        rollout = torch.matmul(rollout, attn_heads_mean)
    return rollout

# ========= Forward Pass =========
with torch.no_grad():
    _, attn_weights = model(input_tensor)
    attention_rollout = compute_attention_rollout(attn_weights)

# ========= Visualize Attention Rollout =========
def visualize_attention_rollout(attention_rollout, save_path=None):
    # Attention from CLS token to all patches
    cls_to_patches = attention_rollout[0, 0, 1:]  # Shape = (64,)

    num_patches = int(math.sqrt(cls_to_patches.size(0)))
    if num_patches ** 2 != cls_to_patches.size(0):
        raise ValueError(f"Cannot reshape attention rollout to a square map. Invalid number of patches: {cls_to_patches.size(0)}")

    attention_map = cls_to_patches.reshape(num_patches, num_patches).detach().cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Rollout Map")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved attention rollout map to {save_path}")
    plt.show()



visualize_attention_rollout(attention_rollout, "output_viz_rollout/attention_rollout.png")