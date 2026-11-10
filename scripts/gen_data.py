#!/usr/bin/env python3
"""
Data generation script for index controller benchmark.
Generates various distributions of uint64_t keys for testing.
"""

import argparse
import struct
import random
import numpy as np
from pathlib import Path


def generate_uniform(size, min_val=1, max_val=2**63-1):
    """Generate uniformly distributed keys."""
    return np.random.randint(min_val, max_val, size=size, dtype=np.uint64)


def generate_zipf(size, alpha=1.5, max_val=2**32):
    """Generate Zipfian distributed keys."""
    keys = np.random.zipf(alpha, size).astype(np.uint64)
    keys = (keys * (max_val // keys.max())).astype(np.uint64)
    return keys


def generate_normal(size, mu=None, sigma=None):
    """Generate normally distributed keys."""
    if mu is None:
        mu = 2**32
    if sigma is None:
        sigma = 2**30

    keys = np.random.normal(mu, sigma, size)
    keys = np.clip(keys, 1, 2**63-1).astype(np.uint64)
    return keys


def generate_sequential(size, start=1, step=1):
    """Generate sequential keys."""
    return np.arange(start, start + size * step, step, dtype=np.uint64)


def generate_clustered(size, num_clusters=10, cluster_size_ratio=0.1):
    """Generate clustered keys."""
    keys = []
    keys_per_cluster = size // num_clusters

    centers = np.random.randint(1, 2**63-1, size=num_clusters, dtype=np.uint64)

    for center in centers:
        cluster_spread = int(2**32 * cluster_size_ratio)
        cluster_keys = np.random.randint(
            max(1, center - cluster_spread),
            min(2**63-1, center + cluster_spread),
            size=keys_per_cluster,
            dtype=np.uint64
        )
        keys.extend(cluster_keys)

    remaining = size - len(keys)
    if remaining > 0:
        keys.extend(generate_uniform(remaining))

    return np.array(keys, dtype=np.uint64)


def generate_exponential(size, lam=1e-9):
    """Generate exponentially distributed keys."""
    keys = np.random.exponential(1/lam, size)
    keys = np.clip(keys, 1, 2**63-1).astype(np.uint64)
    return keys


def save_binary(keys, output_path):
    """Save keys in binary format."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('Q', len(keys)))
        f.write(keys.tobytes())
    print(f"Saved {len(keys)} keys to {output_path} (binary format)")


def save_text(keys, output_path):
    """Save keys in text format."""
    with open(output_path, 'w') as f:
        for key in keys:
            f.write(f"{key}\n")
    print(f"Saved {len(keys)} keys to {output_path} (text format)")


def print_statistics(keys):
    """Print statistics about the generated keys."""
    print("\nDataset Statistics:")
    print(f"  Count: {len(keys)}")
    print(f"  Min: {keys.min()}")
    print(f"  Max: {keys.max()}")
    print(f"  Mean: {keys.mean():.2f}")
    print(f"  Median: {np.median(keys):.2f}")
    print(f"  Std Dev: {keys.std():.2f}")
    print(f"  Unique keys: {len(np.unique(keys))} ({100.0*len(np.unique(keys))/len(keys):.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate test data for index benchmark')
    parser.add_argument('--type', type=str, required=True,
                        choices=['uniform', 'zipf', 'normal', 'sequential',
                                 'clustered', 'exponential'],
                        help='Distribution type')
    parser.add_argument('--size', type=int, required=True,
                        help='Number of keys to generate')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--format', type=str, default='binary',
                        choices=['binary', 'text'],
                        help='Output format')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle keys after generation')

    parser.add_argument('--alpha', type=float, default=1.5,
                        help='Alpha parameter for Zipf distribution')
    parser.add_argument('--mu', type=float, default=None,
                        help='Mean for normal distribution')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Std dev for normal distribution')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters for clustered distribution')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"Generating {args.size} keys with {args.type} distribution...")

    if args.type == 'uniform':
        keys = generate_uniform(args.size)
    elif args.type == 'zipf':
        keys = generate_zipf(args.size, alpha=args.alpha)
    elif args.type == 'normal':
        keys = generate_normal(args.size, mu=args.mu, sigma=args.sigma)
    elif args.type == 'sequential':
        keys = generate_sequential(args.size)
    elif args.type == 'clustered':
        keys = generate_clustered(args.size, num_clusters=args.num_clusters)
    elif args.type == 'exponential':
        keys = generate_exponential(args.size)

    if args.shuffle:
        print("Shuffling keys...")
        np.random.shuffle(keys)

    print_statistics(keys)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'binary':
        save_binary(keys, args.output)
    else:
        save_text(keys, args.output)


if __name__ == '__main__':
    main()
