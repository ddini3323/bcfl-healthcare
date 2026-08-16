"""
Hybrid privacy-preserving mechanisms for federated aggregation:

1. Differential Privacy: per-update L2 clipping + calibrated Gaussian noise.
   Clipping bounds each client's sensitivity, which is what makes the
   noise addition a meaningful (epsilon, delta)-DP mechanism rather than
   arbitrary noise.

2. Secure Aggregation: pairwise additive masks that cancel out once every
   client's masked update is summed, so the server only ever reconstructs
   the aggregate -- never an individual client's raw update. This is a
   simplified, single-machine simulation of secure aggregation (real
   deployments use a key-exchange protocol between clients); it demonstrates
   the same information-theoretic guarantee: the server cannot see or
   recover the model of any single client, since the mask can only be
   cancelled out with the mask of another party.

Combining both gives you the "hybrid" mechanism the objectives ask for:
DP bounds what any observer (including the server) can infer about a
single client's data from the aggregate, and secure aggregation hides
each client's individual contribution from the server itself.
"""

import numpy as np

import config


# ----------------------------------------------------------------------
# 1. Differential privacy: clip then noise
# ----------------------------------------------------------------------
def clip_weights(weights, clip_norm=config.DP_CLIP_NORM):
    """Clip the global L2 norm of a client's weight update to `clip_norm`."""
    flat_norm = np.sqrt(sum(np.sum(np.square(w)) for w in weights))
    scale = min(1.0, clip_norm / (flat_norm + 1e-8))
    return [w * scale for w in weights]


def add_gaussian_noise(weights, clip_norm=config.DP_CLIP_NORM,
                        noise_multiplier=config.DP_NOISE_MULTIPLIER):
    """Add Gaussian noise calibrated to the clipping norm. Must be called
    AFTER clip_weights for the noise to correspond to a real DP guarantee."""
    std = noise_multiplier * clip_norm
    return [w + np.random.normal(0, std, size=w.shape).astype(w.dtype) for w in weights]


def apply_differential_privacy(weights, clip_norm=config.DP_CLIP_NORM,
                                noise_multiplier=config.DP_NOISE_MULTIPLIER):
    clipped = clip_weights(weights, clip_norm)
    return add_gaussian_noise(clipped, clip_norm, noise_multiplier)


def estimate_epsilon(num_rounds, num_clients, noise_multiplier=config.DP_NOISE_MULTIPLIER,
                      delta=config.DP_DELTA, sampling_rate=1.0):
    """Rough (epsilon, delta)-DP estimate using the simple Gaussian
    mechanism composition bound: epsilon ~= sqrt(2 * rounds * ln(1/delta)) / noise_multiplier.
    This is a coarse advanced-composition estimate for reporting purposes --
    for a rigorous, tight accountant use the `opacus` (PyTorch) or
    `tensorflow-privacy` RDP accountant instead.
    """
    if noise_multiplier <= 0:
        return float('inf')
    epsilon = (np.sqrt(2 * num_rounds * np.log(1 / delta)) / noise_multiplier) * sampling_rate
    return float(epsilon)


# ----------------------------------------------------------------------
# 2. Secure aggregation via pairwise cancelling masks
# ----------------------------------------------------------------------
def generate_pairwise_masks(num_clients, weight_shapes, seed=None):
    """For each pair of clients (i, j), draw a random mask and add it to
    client i's mask set, subtract it from client j's. When every client's
    masked update is summed, all pairwise masks cancel out exactly,
    leaving only the true sum of updates."""
    rng = np.random.default_rng(seed)
    masks = {i: [np.zeros(s, dtype='float32') for s in weight_shapes] for i in range(num_clients)}
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            for k, shape in enumerate(weight_shapes):
                m = rng.normal(0, 1, size=shape).astype('float32')
                masks[i][k] += m
                masks[j][k] -= m
    return masks


def mask_weights(weights, mask):
    return [w + m for w, m in zip(weights, mask)]


# ----------------------------------------------------------------------
# Combined pipeline
# ----------------------------------------------------------------------
def apply_hybrid_privacy(client_weights_list, use_secure_agg=config.USE_SECURE_AGGREGATION,
                          clip_norm=config.DP_CLIP_NORM, noise_multiplier=config.DP_NOISE_MULTIPLIER,
                          seed=config.RANDOM_SEED):
    """Apply DP (clip + noise) to every client's update, then optionally
    layer secure-aggregation masks on top before the updates are sent to
    the server for averaging."""
    dp_weights = [apply_differential_privacy(w, clip_norm, noise_multiplier)
                  for w in client_weights_list]

    if not use_secure_agg:
        return dp_weights

    shapes = [w.shape for w in dp_weights[0]]
    masks = generate_pairwise_masks(len(dp_weights), shapes, seed=seed)
    masked = [mask_weights(dp_weights[i], masks[i]) for i in range(len(dp_weights))]
    return masked
