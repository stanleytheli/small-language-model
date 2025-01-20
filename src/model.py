import torch
import torch.nn as nn
from torch.nn import functional as F
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_filepath = "./data/tinyshakespeare.txt"

# hyperparams
proportion_train = 0.9

batch_size = 64
context_len = 256
learning_rate = 3e-4

epochs = 20000
epoch_interval = 500

# IMPORTANT: n_embed MUST be divisible by n_heads !
n_heads = 8
n_layers = 10
n_embed = 64

dropout = 0.2

# fetch data
with open(data_filepath, "r", encoding="utf-8") as f:
    text = f.read()

# tokenize data
chars = sorted(list(set(text)))
vocab_size = len(chars)

s_to_i = {ch : i for i, ch in enumerate(chars)}
i_to_s = {i : ch for i, ch in enumerate(chars)}

def encode(s):
    return [s_to_i[ch] for ch in s]
def decode(l):
    return ''.join([i_to_s[i] for i in l])

# split data into train and val groups
data = torch.tensor(encode(text), dtype = torch.long)
n = int(proportion_train * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    return x, y

# simple, pretty bad model
class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(context_len, n_embed)

        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.lnorm_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        # during training, B = batch_size, T = context_len
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B, T, n_embed)
        position_embeddings = self.position_embedding_table(torch.arange(T)) # (T, n_embed)
        embeddings = token_embeddings + position_embeddings # (B, T, n_embed)

        # Go through the blocks
        embeddings = self.blocks(embeddings)
        # Final LayerNorm
        embeddings = self.lnorm_final(embeddings)
        # Unembedding
        logits = self.lm_head(embeddings) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # as a note: B = batch size, T = context length, C = vocab size
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of current context
        for _ in range(max_new_tokens):
            # Cropping to get the last context_len tokens
            idx_cond = idx[:, -context_len:] 
            # get predictions
            logits, loss = self(idx_cond) # logits: (B, T, C)
            # convert last logit to probabilities
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # add that to our generation
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute affinities
        weights = (q @ k.transpose(-2, -1)) / (C ** 0.5) # (B, T, C) @ (B, C, T) --> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf")) 
        weights = F.softmax(weights, dim=-1) 
        weights = self.dropout(weights)

        v = self.value(x) # (B, T, C)
        output = weights @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for i in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

# perceptron
class FeedForward(nn.Module):
    def __init__(self, n_embed, n_inner_layer):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_embed, n_inner_layer),
            nn.ReLU(),
            nn.Linear(n_inner_layer, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.feedfwd = FeedForward(n_embed, 4 * n_embed)

        self.lnorm_1 = nn.LayerNorm(n_embed)
        self.lnorm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_heads(self.lnorm_1(x))
        x = x + self.feedfwd(self.lnorm_2(x))
        return x

def validate(model, sample_size = 10):
    model.eval()

    n_seen = 0
    total_loss = 0

    for i in range(sample_size):
        xb, yb = get_batch("val")
        logits, loss = model(xb, yb)

        n_seen += 1
        total_loss += loss.item()

    return total_loss / n_seen

model = LanguageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# training!
n_seen = 0
total_loss = 0
last_time = time.time()

for epoch in range(epochs):
    model.train()
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    n_seen += 1
    total_loss += loss.item()

    if (epoch + 1) % epoch_interval == 0:
        now = time.time()
        print(f"step {epoch + 1}/{epochs}  took {now - last_time:<4f}s  avg loss: {total_loss / n_seen:<5f} ", end="")
        print(f"val loss: {validate(model):<5f}")
        last_time = now

# generating some text
idx = torch.zeros((1, 1), dtype = torch.long)
generation = model.generate(idx, max_new_tokens=500)[0].tolist()
gen_text = decode(generation)
print(gen_text)

