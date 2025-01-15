from codec_studio import codecs
from tqdm.notebook import tqdm

WS = 8
EPOCH = 8
RANGE_INIT = 0.3
RANGE_LIMIT = 0.05

SEED = 42
np.random.seed(SEED)

# wavelet = codecs.wavelet_codec()
decomposed_numpy = process_audio(train_data, WS).numpy()
total_samples = len(decomposed_numpy)
tree = CortexTree(window_size=WS, range_init=RANGE_INIT, range_limit=RANGE_LIMIT)
for e in tqdm(range(EPOCH), desc="Epoch"):
    added = 0
    leaf = 0
    max_depth = 0
    for d_i, d in enumerate(decomposed_numpy): # tqdm(decomposed_numpy, desc="Data", leave=False, mininterval=1):
            # d += np.random.normal(0, 1/(2**15), len(d))
            depth, new_added, new_leaf = tree.train_single(d)
            added += new_added
            leaf += new_leaf
            max_depth = max(max_depth, depth)
            if d_i % 1000 == 0:
                print(f"\rEpoch: {e} - Progress: {100 * (d_i/total_samples):.2f}%", end="")
    if added > 0 or leaf > 0:
        print(f"\rEpoch: {e} - Depth: {max_depth}, Added: {added}, Leaf: {leaf}")

print(f"Epoch: {e}, Depth: {max_depth}, Added: {added}, Leaf: {leaf}")

cb = tree.complete()
# print(f"Codebook: {cb.paths}")
print(f"Codebook shape: {cb.paths.shape}")



from codec_studio import codecs
from tqdm.notebook import tqdm

EPOCH = 4
RANGE_INIT = 0.1
RANGE_LIMIT = 0.002

SEED = 42
np.random.seed(SEED)

# windowing = codecs.window_codec(WS)
# wavelet = codecs.wavelet_codec()
all_codebook_values = cb.paths.reshape(-1,1) #(-1)
total_samples = len(all_codebook_values)
cdbk_tree = CortexTree(window_size=1, range_init=RANGE_INIT, range_limit=RANGE_LIMIT)
for e in tqdm(range(EPOCH), desc="Epoch"):
    added = 0
    leaf = 0
    max_depth = 0
    for d_i, d in enumerate(all_codebook_values): # tqdm(decomposed_numpy, desc="Data", leave=False, mininterval=1):
            # d += np.random.normal(0, 1/(2**15), len(d))
            depth, new_added, new_leaf = cdbk_tree.train_single(d)
            added += new_added
            leaf += new_leaf
            max_depth = max(max_depth, depth)
            if d_i % 1000 == 0:
                print(f"\rEpoch: {e} - Progress: {100 * (d_i/total_samples):.2f}%", end="")
    if added > 0 or leaf > 0:
        print(f"\rEpoch: {e} - Depth: {max_depth}, Added: {added}, Leaf: {leaf}")

print(f"Epoch: {e}, Depth: {max_depth}, Added: {added}, Leaf: {leaf}")

cdbk_cb = cdbk_tree.complete()
# print(f"Codebook: {cb.paths}")
print(f"Codebook shape: {cdbk_cb.paths.shape}")
# print(np.unique(cdbk_cb.encode(all_codebook_values)).shape)