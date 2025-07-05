import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import math
import numpy as np
from torchvision import transforms
from einops import rearrange
import os

# ========= Set device =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
print("Using device:", device)

# ========= Define Modules =========
class DifferentialAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1, layer_idx=1, total_layers=12):
        super(DifferentialAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // (2 * num_heads)
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(model_dim, model_dim * 2)
        self.key = nn.Linear(model_dim, model_dim * 2)
        self.value = nn.Linear(model_dim, model_dim)  # Changed to model_dim instead of model_dim*2
        self.out = nn.Linear(model_dim, model_dim)

        self.lambda_q1 = nn.Parameter(torch.randn(self.head_dim) * 0.05)
        self.lambda_k1 = nn.Parameter(torch.randn(self.head_dim) * 0.05)
        self.lambda_q2 = nn.Parameter(torch.randn(self.head_dim) * 0.05)
        self.lambda_k2 = nn.Parameter(torch.randn(self.head_dim) * 0.05)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (layer_idx - 1))
        # self.lambda_init = 0.9
        self.norm = nn.GroupNorm(num_groups=num_heads, num_channels=model_dim)
        self.attn_weights = None # Initialize attn_weights attribute

    def compute_lambda(self):
        lambda_val = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1)) - \
                     torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2)) + \
                     self.lambda_init
        return lambda_val

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, -1)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, dim_per_head]

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, seq_len, -1)

    def differential_attention(self, Q1, Q2, K1, K2, V):
        s = 1.0 / math.sqrt(self.head_dim)

        A1 = torch.matmul(Q1, K1.transpose(-2, -1)) * s
        A2 = torch.matmul(Q2, K2.transpose(-2, -1)) * s

        A1 = torch.softmax(A1, dim=-1)
        A2 = torch.softmax(A2, dim=-1)

        A1 = self.dropout(A1)
        A2 = self.dropout(A2)

        lambda_val = self.compute_lambda()

        diff_attn = A1 - lambda_val * A2
        self.attn_weights = diff_attn # Store attention weights here

        output = torch.matmul(diff_attn, V)
        return output

    def forward(self, X):
        batch_size, seq_len, _ = X.size()

        Q = self.query(X)  # [batch, seq_len, 2*model_dim]
        K = self.key(X)    # [batch, seq_len, 2*model_dim]
        V = self.value(X)  # [batch, seq_len, model_dim]

        Q = self.split_heads(Q)  # [batch, heads, seq_len, 2*head_dim]
        K = self.split_heads(K)  # [batch, heads, seq_len, 2*head_dim]
        V = self.split_heads(V)  # [batch, heads, seq_len, head_dim]

        Q1, Q2 = torch.chunk(Q, 2, dim=-1)  # [batch, heads, seq_len, head_dim]
        K1, K2 = torch.chunk(K, 2, dim=-1)  # [batch, heads, seq_len, head_dim]

        output = self.differential_attention(Q1, Q2, K1, K2, V)

        output = self.combine_heads(output)  # [batch, seq_len, model_dim]

        # Apply group normalization
        output = self.norm(output.transpose(1, 2)).transpose(1, 2)

        return self.out(output)


class FeedForward(nn.Module):
    def __init__(self, model_dim, hid_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(model_dim, hid_dim)
        self.ac1 = nn.GELU()
        self.l2 = nn.Linear(hid_dim, model_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self, inp):
        inp = self.l1(inp)
        inp = self.dropout(self.ac1(inp))
        inp = self.l2(inp)
        inp = self.dropout(inp)
        return inp


class Encoder(nn.Module):
    def __init__(self, model_dim, num_heads, hid_dim, dropout, layer_idx=1, total_layers=12):
        super(Encoder, self).__init__()
        self.self_attn = DifferentialAttention(model_dim, num_heads, dropout, layer_idx, total_layers)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        normed_inp = self.norm1(inp)
        att_output = self.self_attn(normed_inp)
        inp = inp + self.dropout(att_output)

        normed_inp = self.norm2(inp)
        ffn_out = self.ffn(normed_inp)
        inp = inp + self.dropout(ffn_out)

        return inp


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
        # Ensure H and W are divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Image dimensions ({H}, {W}) must be divisible by patch size ({self.patch_size}).")

        # Use einops for clarity (optional but recommended)
        # x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        # Manual reshape and permute
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()  # (B, H/ps, W/ps, C, ps, ps)
        x = x.view(B, self.num_patches, self.patch_dim)  # (B, num_patches, patch_dim)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, model_dim, num_layer, num_heads, hid_dim, dropout=0.1):
        super(Transformer, self).__init__()

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, model_dim)
        self.num_patches = self.patch_embed.num_patches

        # Class token and positional encoding
        self.cls_embed = nn.Parameter(torch.randn(1, 1, model_dim))
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_patches + 1, model_dim))

        # Encoder layers
        self.encoders = nn.ModuleList([
            Encoder(model_dim, num_heads, hid_dim, dropout, layer_idx=i+1, total_layers=num_layer)
            for i in range(num_layer)
        ])

    def forward(self, src):
        patch_emb = self.patch_embed(src)

        B = patch_emb.size(0)
        cls_tokens = self.cls_embed.expand(B, -1, -1)
        src_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        src_emb = src_emb + self.pos_enc

        for encoder in self.encoders:
            src_emb = encoder(src_emb)

        return src_emb[:, 0]  # Class token



class ViT(nn.Module):

    def __init__(self, img_size, patch_size, in_channels, num_classes, model_dim=768, num_layer=12, num_heads=12, hid_dim=3072, dropout=0.1):
        super(ViT, self).__init__()
        self.transformer = Transformer(img_size, patch_size, in_channels, model_dim, num_layer, num_heads, hid_dim, dropout)
        self.norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = self.norm(x)
        x = self.fc(x)
        return x


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
# Ensure the model file exists
model_path = "../80percent_above.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device)) # Add map_location
model = model.to(device)
model.eval()

# ========= Hook: Extract attention maps =========
attentions = [] # Initialize/clear the list here

def attention_hook(module, input, output):
    # Shape: (B, heads, tokens, tokens)
    # Check if attn_weights exists and is not None
    if hasattr(module, "attn_weights") and module.attn_weights is not None:
        attentions.append(module.attn_weights.detach().cpu())
    else:
        print(f"Warning: Attention weights not found or None in module {module}")


# Register hooks *after* initializing the list
for encoder in model.transformer.encoders:
    encoder.self_attn.register_forward_hook(attention_hook)

# ========= Load a sample image from CIFAR-10 test set =========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
cifar10 = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
image, label = cifar10[0]
input_tensor = image.unsqueeze(0).to(device)

# ========= Forward Pass =========
attentions = [] # Clear attentions list *before* the forward pass
with torch.no_grad():
    _ = model(input_tensor)

# Check if attentions list is empty before stacking
if not attentions:
    print("Error: No attention weights were captured. Check hook registration and MultiHeadAttention implementation.")
    exit() # Exit if no attentions were captured

attentions = torch.stack(attentions)  # [num_layers, B, heads, tokens, tokens]
attentions = attentions.squeeze(1)    # [num_layers, heads, tokens, tokens]
num_layers, num_heads, tokens, _ = attentions.shape
print(f"Captured attentions shape: {attentions.shape}") # Debug print


# ========= Visualization Functions =========
def visualize_attention_map(attn, title, save_path=None):
    # attn shape: [heads, tokens, tokens]
    num_heads_in_tensor = attn.shape[0] # Get number of heads from the tensor itself
    cls_attn = attn[:, 0, 1:] # Attention from CLS token to patch tokens [heads, num_patches]
    num_patches = cls_attn.shape[-1]
    # Calculate grid size (e.g., 8x8 for 64 patches)
    size = int(np.sqrt(num_patches))
    if size * size != num_patches:
        print(f"Warning: Number of patches ({num_patches}) is not a perfect square. Visualization might be incorrect.")
        # Handle non-square cases if necessary, e.g., padding or different layout
        return # Or adjust visualization logic

    # Use num_heads_in_tensor for subplot creation
    fig, axs = plt.subplots(1, num_heads_in_tensor, figsize=(3*num_heads_in_tensor, 3.5)) # Adjust figsize
    fig.suptitle(title, fontsize=16) # Add main title

    # Handle case where num_heads_in_tensor is 1
    if num_heads_in_tensor == 1:
        axs = [axs]

    # Use num_heads_in_tensor for the loop
    for i in range(num_heads_in_tensor):
        im = cls_attn[i].reshape(size, size).numpy()
        axs[i].imshow(im, cmap='viridis')
        axs[i].set_title(f'Head {i+1}')
        axs[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    if save_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(save_path)
        if output_dir: # Ensure dirname is not empty (e.g., if saving in current dir)
             os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved attention map to {save_path}")
    # plt.show() # Comment out or remove plt.show() if saving figures in a loop
    # plt.savefig("viz.png") # Removed redundant save with fixed name
    plt.close(fig) # Close the figure after showing/saving


def visualize_avg_attention(attn, title, save_path=None):
    # attn shape: [heads, tokens, tokens]
    avg_attn = attn.mean(0, keepdim=True)  # Average over heads -> [1, tokens, tokens]
    # Pass the averaged attention map (which has 1 head) to the visualization function
    visualize_attention_map(avg_attn, title, save_path)

# ========= Visualize Last Layer Attention Heads =========
print("\nVisualizing Last Layer Attention (All Heads)...")
visualize_attention_map(attentions[-1], "Last Layer Attention (All Heads)", "output_viz/last_layer_all_heads.png")

# ========= Visualize Last Layer Avg Attention =========
print("\nVisualizing Last Layer Average Attention...")
visualize_avg_attention(attentions[-1], "Last Layer Avg Attention", "output_viz/last_layer_avg.png")

# ========= Visualize All Layers Avg Attention =========
print("\nVisualizing Average Attention for Each Layer...")
for l in range(num_layers):
    print(f"Layer {l+1}/{num_layers} (Avg Head)")
    visualize_avg_attention(attentions[l], f"Layer {l+1} Avg Attention", f"output_viz/layer_{l+1}_avg.png")

# ========= Visualize All Layers All Heads Attention =========
print("\nVisualizing Attention for Each Head in Each Layer...")
for l in range(num_layers):
    print(f"Layer {l+1}/{num_layers} (All Heads)")
    visualize_attention_map(attentions[l], f"Layer {l+1} Attention (All Heads)", f"output_viz/layer_{l+1}_all_heads.png")


print("\nVisualization complete.")