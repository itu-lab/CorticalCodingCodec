# corticod Library Documentation

## Overview
`corticod` is a biomimetic cortical coding library designed for hierarchical data processing, pattern recognition, and sequential data compression. It models a neural-inspired tree structure for adaptive learning and sparse representation of data.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/itu-lab/CorticalCodingCodec.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dependencies
- Python >= 3.8
- NumPy
- scikit-learn
- PyTorch (optional, for GPU support)

---

## Modules

### **1. `tree.py`**
Defines the hierarchical tree and node structure.

#### **Classes**

- **`Node`**:
  Represents a single tree node.
  - **Attributes**:
    - `parent`: Parent node.
    - `level`: Depth of the node in the tree.
    - `children`: Managed by the `NodeChildren` class.
    - `data`: Value associated with the node.
    - `maturity`: Indicates if the node has matured.
  - **Key Methods**:
    - `get_data()`: Get the node's data.
    - `set_data(data)`: Set the node's data.
    - `is_mature()`: Check if the node is mature.
    - `add_child(data)`: Add a child node.

- **`NodeChildren`**:
  Manages the children of a node.
  - **Attributes**:
    - `owner`: Parent node.
    - `data_vector`: Stores children’s data in sorted order.
    - `maturity_mask`: Indicates which children are mature.
  - **Key Methods**:
    - `add_child(data)`: Add a child and maintain sorted order.
    - `remove_child(index)`: Remove a child by index.
    - `find_index(data)`: Find the insertion index for new data.

- **`Tree`**:
  Represents the entire tree structure.
  - **Attributes**:
    - `root`: Root node of the tree.
  - **Key Methods**:
    - `paths(window_size)`: Retrieve paths from root to mature leaf nodes of fixed length.

---

### **2. `cortical_tree.py`**
Implements the cortical coding algorithm.

#### **Classes**

- **`CorticalNode`**:
  Extends the `Node` class with biomimetic properties.
  - **Attributes**:
    - `maturation_energy`: Tracks node's maturity progress.
    - `range`: Range threshold for matching inputs.
  - **Key Methods**:
    - `update(data, range_limit)`: Updates the node with new input data.
    - `find_closest_child(data)`: Finds the closest matching child node.
    - `get_total_progeny()`: Counts all descendant nodes.

- **`CortexTree`**:
  Extends the `Tree` class with cortical coding logic.
  - **Attributes**:
    - `window_size`: Length of the input data window.
    - `range_limit`: Minimum range for node matching.
  - **Key Methods**:
    - `closest_path(wave)`: Finds the closest path in the tree for a given input wave.
    - `train_single(wave)`: Trains the tree with a single input wave.
    - `train(waves, epochs)`: Trains the tree with multiple input waves over several epochs.
    - `complete(paths)`: Creates a codebook from paths.

---

### **3. `codebook.py`**
Handles encoding and decoding of data using codebooks generated from the tree paths.

- **`Codebook` Class**:
  - Provides methods for signal quantization, encoding, and decoding.
  - Supports both batch and single-signal operations.
  - **Key Methods**:
    - `encode_all(waves: np.ndarray) -> np.ndarray`: Encodes all input waves into their respective paths.
    - `decode(indices: np.ndarray) -> np.ndarray`: Decodes indices back into waveforms using the codebook.

---

## Usage

### **Example: Training the Tree**
```python
import numpy as np
from corticod import CortexTree

# Generate dummy data
data = np.random.rand(1000, 8)  # 1000 waves, each of length 8

# Initialize CortexTree
tree = CortexTree(window_size=8, range_init=50, range_limit=10)

# Train the tree
tree.train(data, epochs=10)

# Find the closest path for a new wave
query_wave = np.random.rand(8)
result = tree.closest_path(query_wave)
print(f"Closest Path: {result}")

# Get codebook, to store/load
codebook = tree.complete()

# Encode all waves
encoded = codebook.encode_all(data)

# Decode back to approximate waves
decoded = codebook.decode(encoded)

```

---

## API Reference

### **`Node` Class**
- `get_data() -> Any`: Returns the node's data.
- `set_data(data: Any)`: Sets the node's data.
- `is_mature() -> bool`: Checks if the node has matured.
- `add_child(data: Any) -> Node`: Adds a new child with the given data.

### **`CortexTree` Class**
- `train(waves: np.ndarray, epochs: int) -> Tuple[int, int]`:
  Trains the tree using the input data.
  - **Parameters**:
    - `waves`: Input data, each row representing a wave.
    - `epochs`: Number of training iterations.
  - **Returns**: Total nodes added and leaf nodes matured.

- `closest_path(wave: np.ndarray) -> np.ndarray`:
  Finds the closest matching path for the given wave.

### **`Codebook` Class**
- `encode_all(waves: np.ndarray) -> np.ndarray`:
  Encodes all input waves into their respective paths.

- `decode(indices: np.ndarray) -> np.ndarray`:
  Decodes indices back into waveforms.

---

## Utilities

The `corticod.utils` module includes helper functions for audio preprocessing and feature extraction. These utilities are essential for preparing raw audio data before applying the `CortexTree` algorithm.

#### Key Utility Functions

1. **Audio Loading**:
   - `audio.get_audio_data(audio_path)`: Loads audio from a file and extracts metadata such as sample rate, bit depth, and raw audio data.

2. **Audio Preprocessing**:
   - `preprocessing.process_audio(data, window_size)`: Splits raw audio into fixed-size windows and applies Haar wavelet decomposition for feature extraction.
   - `preprocessing.process_audio_inverse(data)`: Reconstructs the full audio signal from segmented wavelet coefficients using Haar wavelet reconstruction.

#### Workflow Description

1. **Load Raw Audio**:
   Use `get_audio_data` to read audio files and retrieve the necessary metadata for downstream processing.

2. **Segment Audio into Windows**:
   Apply `process_audio` to convert raw audio signals into windows of a specified size (`window_size`), preparing the data for hierarchical tree encoding.

3. **Reconstruct Audio**:
   After decoding the audio with `CortexTree`, use `process_audio_inverse` to reconstruct the signal back to its original form.

For more information read the corresponding [README](./corticod/utils/README.md) file under the corticod/utils directory. 

---

## Tests

### **Basic Functionality**
- Ensures nodes are added, updated, and matured correctly.
- Validates traversal and matching logic.

### **Performance Tests**
- Measures speed and memory usage for large datasets.
- Tests scalability with high-depth trees.

### **Works for Edge Cases**
- Empty datasets.
- Extremely noisy input data.
- Repeated inputs with minimal variance.

---

## Future Improvements
1. Parallelization for large datasets.
2. Optimized storage for deeper trees.
3. Integration with visualization tools for tree paths and node relationships.
