data_filepath = "./data/article.txt"

# vocab size starts at 256 (bytes)
vocab_size = 1000
num_merges = vocab_size - 256

def count_pairs(tokens):
    # Count pair occurrences
    pairs = {}
    for i in range(len(tokens) - 1):
        if (tokens[i], tokens[i + 1]) in pairs:
            pairs[(tokens[i], tokens[i + 1])] += 1
        else:
            pairs[(tokens[i], tokens[i + 1])] = 1
    return pairs

def merge(tokens, replace_pair, new_id):
    new_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == replace_pair:
            new_tokens.append(new_id)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens

merges = {} # pair --> id

# grab data
with open(data_filepath, "r", encoding="utf-8") as f:
    text = f.read()

# raw utf-8 tokenization 
tokens = list(map(int, text.encode("utf-8")))

# byte pair encoding step
for i in range(num_merges):
    new_id = 256 + i
    
    pairs = count_pairs(tokens)
    max_pair = max(pairs, key=pairs.get)

    merges[max_pair] = new_id
    tokens = merge(tokens, max_pair, new_id)

# the data in merges fully characterizes the tokenizer!

vocab = {id: bytes([id]) for id in range(256)} # id --> byte
for (pair_0, pair_1), id in merges.items(): 
    # iteration is guaranteed to be in order of insertion
    # breaks down all merges into their byte sequences
    vocab[id] = vocab[pair_0] + vocab[pair_1]

def decode(tokens):
    token_bytes = b"".join(vocab[id] for id in tokens)
    text = token_bytes.decode("utf-8", errors="replace")
    return text

def encode(text):
    tokens = text.encode("utf-8")
    for pair, pair_id in merges.items(): # maybe more efficient way to do this?
        tokens = merge(tokens, pair, pair_id)
    return tokens

test_string = "The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks)."
print([decode([token]) for token in encode(test_string)])