import functools
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

class Util():
    @staticmethod
    def handle_list_input(original_func=None, func_2D=None, func_tensor=None):
        def _decorate(func):
            @functools.wraps(func)
            def wrapper(self, signal):
                if isinstance(signal, torch.Tensor):
                    if func_tensor is not None:
                        return func_tensor(self, signal)
                    else:
                        signal = signal.numpy()
                elif not isinstance(signal, np.ndarray):
                    signal = np.asarray(signal)
                    # if signal.ndim == 0:
                    #     signal = signal.reshape(1)

                if signal.ndim == 1 or signal.ndim == 0:
                    return func(self, signal)
                elif signal.ndim == 2:
                    if func_2D is None:
                        return np.asarray([func(self, s) for s in signal])
                    else:
                        return func_2D(self, signal)
                else:
                    raise ValueError(f"Input signal must be 1D or 2D array, got {signal.ndim}D array") 
            return wrapper

        if original_func:
            return _decorate(original_func)

        return _decorate

class Codebook():
    def __init__(self, paths, weights=None) -> None:
        if weights is not None:
            self.weights = np.asarray(weights, dtype=np.float32)
        else:
            block_size = paths[0].shape[0]
            self.weights = np.ones(block_size, dtype=np.float32)
        self.paths = paths
        self.paths_tensor = torch.tensor(self.paths).float()

    def __getitem__(self, i):
        return self.paths[i]
    
    def __len__(self):
        return len(self.paths)
    
    def __repr__(self):
        return f"Codebook with {len(self)} paths"

    @Util.handle_list_input
    def distance(self, signal):
        dist = np.linalg.norm(self.paths - signal, axis=1)
        return np.min(dist)

    @Util.handle_list_input
    def quantize(self, signal):
        return self.decode(self.encode(signal))

    @Util.handle_list_input
    def cosine_sim(self, signal):
        quantized = self.quantize(signal)
        return cosine_similarity(quantized.reshape(1, -1), signal.reshape(1, -1))[0][0]

    def encode_2d(self, signals):
        signals *= self.weights
        scaled_paths = self.paths * self.weights
        return np.argmin(np.sum(scaled_paths**2, axis=1, keepdims=True) - 2*np.dot(scaled_paths, signals.T) + np.sum(signals.T**2, axis=0, keepdims=True), axis=0)

    def encode_tensor_batch(self, signals:torch.Tensor, batch_size:int=1024*8):
        # Example of batch processing
        nearest_indices = []
        signals *= self.weights
        scaled_paths_tensor = self.paths_tensor * self.weights

        for i in range(0, signals.shape[0], batch_size):
            end = i + batch_size
            batch_distances = torch.cdist(scaled_paths_tensor, signals[i:end])
            batch_indices = torch.argmin(batch_distances, axis=0)
            # nearest_indices.append(batch_indices + i)  # Correct indices for batch offset
            nearest_indices.append(batch_indices)  # Correct indices for batch offset

        # remaining = signals.shape[0] % batch_size
        # if remaining > 0:
        #     batch_distances = torch.cdist(scaled_paths_tensor, signals[-remaining:])
        #     batch_indices = torch.argmin(batch_distances, axis=0)
        #     nearest_indices.append(batch_indices)

        nearest_indices = torch.cat(nearest_indices)
        return nearest_indices

    def encode_tensor(self, signals:torch.Tensor):
        gpu_usage, gpu_available = torch.cuda.mem_get_info()
        gpu_free = (gpu_available-gpu_usage) # in Bytes
        element_size = max(signals.element_size(), self.paths_tensor.element_size())
        signal_bytes = element_size * signals.nelement()
        path_bytes = element_size * self.paths_tensor.nelement()
        gpu_free -= signal_bytes + path_bytes
        matmul_bytes = signals.shape[0] * self.paths_tensor.shape[0] * element_size
        # if max(signal_bytes, path_bytes, matmul_bytes) > 1 * 1024 * 1024 * 1024: # > 1GB
        # print(f"Signal bytes: {signal_bytes}, Path bytes: {path_bytes}, Matmul bytes: {matmul_bytes}, GPU Free: {gpu_free}")
        if matmul_bytes > gpu_free:
            max_batch_size = int(gpu_free / (self.paths_tensor.shape[0] * element_size))
            batch_size = 2**int(np.log2(max_batch_size))
            # print(f"Decided to batch process - Batch size: {batch_size}")
            return self.encode_tensor_batch(signals, batch_size)
        else:
            signals *= self.weights
            scaled_paths_tensor = self.paths_tensor * self.weights
            return torch.argmin(torch.sum(scaled_paths_tensor**2, axis=1, keepdims=True) - 2*torch.matmul(scaled_paths_tensor, signals.T) + torch.sum(signals.T**2, axis=0, keepdims=True), axis=0)

    @Util.handle_list_input(func_2D=encode_2d, func_tensor=encode_tensor)
    def encode(self, signal):
        signal *= self.weights
        scaled_paths = self.paths * self.weights
        # print(signal, scaled_paths)
        # print(np.argmin(np.linalg.norm(scaled_paths - signal, axis=1)))
        return np.argmin(np.linalg.norm(scaled_paths - signal, axis=1))

    def decode_tensor(self, index):
        return self.paths_tensor[index]

    @Util.handle_list_input(func_tensor=decode_tensor)
    def decode(self, index):
        # print(self.paths, index)
        return self.paths[index]
    
    def encode_all(self, data):
        encoded = []
        for w in data:
            encoded.append(codebook.encode(w))
        encoded = np.asarray(encoded)


if __name__ == "__main__":

    # Example paths for the codebook (2D arrays)
    # paths = [
    #     np.array([[1.0, 0.0], [0.0, 1.0]]),
    #     np.array([[0.0, 1.0], [1.0, 0.0]]),
    #     np.array([[1.0, 1.0], [0.0, 0.0]]),
    #     np.array([[0.5, 0.5], [0.5, 0.5]])
    # ]

    # # Example paths for the codebook
    paths = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0])
    ]

    # Initialize Codebook
    codebook = Codebook(paths)

    # # Example signals for testing
    signals = np.array([
        [0.9, 0.1, 0.0],
        [0.0, 0.8, 0.2],
        [0.3, 0.3, 0.4],
        [0.99, 0.99, 0.99]
    ])

    # Example 2D signals for testing
    # signals = [
    #     np.array([[0.9, 0.1], [0.1, 0.9]]),
    #     np.array([[0.0, 0.8], [0.2, 0.0]]),
    #     np.array([[0.3, 0.3], [0.4, 0.4]]),
    #     np.array([[1.0, 1.0], [0.0, 0.0]])
    # ]

    print("Testing Codebook Operations:\n")

    # Test distance function
    for signal in signals:
        dist = codebook.distance(signal)
        print(f"Distance for {signal}: {dist}")

    # Test quantize function
    for signal in signals:
        quantized_signal = codebook.quantize(signal)
        print(f"Quantized {signal} to {quantized_signal}")

    # Test cosine similarity
    for signal in signals:
        similarity = codebook.cosine_sim(signal)
        print(f"Cosine similarity for {signal}: {similarity}")

    # Test encode and decode
    for signal in signals:
        encoded_index = codebook.encode(signal)
        decoded_signal = codebook.decode(encoded_index)
        print(f"Encoded {signal} to index {encoded_index}, Decoded to {decoded_signal}")

    # Test tensor operations (batch encoding)
    signals_tensor = torch.tensor(signals).float()
    encoded_indices = codebook.encode(signals_tensor)
    decoded_signals = codebook.decode(encoded_indices)

    print("\nTensor Operations:")
    print(f"Encoded Indices: {encoded_indices}")
    print(f"Decoded Signals: {decoded_signals}")

