import numpy as np
from corticod.algorithm import CortexTree

def main():
    # Initialize CortexTree
    cortex_tree = CortexTree(window_size=2, range_init=2, range_limit=1.0)

    # Example wave signals for training
    wave_signals = [
        np.array([1.0, 2.0]),
        np.array([1.5, 2.5]), 
        np.array([2.0, 3.0]),
        np.array([12, 13]),
        np.array([11, 12]),
        np.array([10, 11]),
    ] * 100
    # Compare this with C++ codes

    print("Training CortexTree with wave signals:\n")

    # Train the CortexTree with each wave signal
    for wave in wave_signals:
        levels, added, leafs = cortex_tree.train_single(wave)
        # print(f"Wave: {wave}")
        # print(f"Levels Traversed: {levels}, Nodes Added: {added}, Leafs: {leafs}\n")

    # All paths
    print('Paths:')
    print(cortex_tree.paths(2), "\n")

    # Test closest_path function
    test_wave = np.array([1.0, 3.0])
    closest_path = cortex_tree.closest_path(test_wave)
    print(f"Closest Path for wave {test_wave}: {closest_path}\n")

    # Test closest_full_path function
    full_path_distance = cortex_tree.closest_full_path(test_wave)
    print(f"Closest Full Path Distance for wave {test_wave}: {full_path_distance}\n")

if __name__ == "__main__":
    main()