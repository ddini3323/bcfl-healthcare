"""
Server-side weighted aggregation of (already privacy-protected) client
weight updates.
"""

import numpy as np


def weighted_average(client_weights_list, client_sizes):
    """Weighted FedAvg: each client's contribution is weighted by how
    many local samples it trained on."""
    total = sum(client_sizes)
    normalized = [s / total for s in client_sizes]

    n_layers = len(client_weights_list[0])
    aggregated = [np.zeros_like(w) for w in client_weights_list[0]]

    for weights, weight_frac in zip(client_weights_list, normalized):
        for i in range(n_layers):
            aggregated[i] += weight_frac * weights[i]

    return aggregated
