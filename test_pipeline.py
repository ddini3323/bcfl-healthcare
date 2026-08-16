"""
Smoke test: runs the full FL + blockchain + privacy + evaluation pipeline
on small random synthetic data (no NIH dataset download required). Use
this first to confirm your local environment / VS Code setup is correct
before pointing config.py at the real dataset.

Usage:
    python test_pipeline.py
"""

import numpy as np
import tensorflow as tf

from blockchain.ledger import Blockchain
from privacy.mechanisms import apply_hybrid_privacy, estimate_epsilon
from federated.server import weighted_average
from models.model import build_lightweight_model


def make_synthetic_batch(n=16, img_size=(224, 224, 3), num_classes=14):
    x = np.random.rand(n, *img_size).astype('float32')
    y = (np.random.rand(n, num_classes) > 0.8).astype('float32')
    return x, y


def main():
    print("1. Building lightweight model ...")
    num_classes = 14
    model = build_lightweight_model(num_classes=num_classes)
    global_weights = model.get_weights()
    print(f"   OK -- {len(global_weights)} weight tensors")

    print("\n2. Simulating 3 clients training locally ...")
    client_weights = []
    for i in range(3):
        client_model = tf.keras.models.clone_model(model)
        client_model.set_weights(global_weights)
        client_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        x, y = make_synthetic_batch(n=16, num_classes=num_classes)
        client_model.fit(x, y, epochs=1, verbose=0)
        client_weights.append(client_model.get_weights())
    print(f"   OK -- trained {len(client_weights)} clients")

    print("\n3. Applying hybrid privacy (clip + DP noise + secure-agg masks) ...")
    protected = apply_hybrid_privacy(client_weights)
    print("   OK -- weights are clipped, noised, and masked")

    print("\n4. Logging updates to blockchain ...")
    ledger = Blockchain()
    for i, w in enumerate(protected):
        ledger.add_block(f"client_{i}", Blockchain.hash_weights(w), round_num=1)
    valid, msg = ledger.is_valid()
    print(f"   Chain valid before tampering: {valid} ({msg})")

    print("\n5. Demonstrating tamper detection ...")
    ledger.chain[1].client_name = "attacker"  # mutate a block's content directly
    valid, msg = ledger.is_valid()
    print(f"   Chain valid after tampering block 1: {valid} ({msg})")
    assert not valid, "Tamper detection failed -- this should have been caught!"
    print("   Tamper detection works correctly.")

    print("\n6. Aggregating client updates (weighted FedAvg) ...")
    aggregated = weighted_average(protected, client_sizes=[16, 16, 16])
    print(f"   OK -- aggregated {len(aggregated)} weight tensors")

    print("\n7. Estimating DP privacy budget ...")
    eps = estimate_epsilon(num_rounds=5, num_clients=3)
    print(f"   epsilon ~= {eps:.2f}")

    print("\nAll pipeline components work correctly.")
    print("Next step: point config.py at the real NIH ChestX-ray14 dataset and run main.py")


if __name__ == '__main__':
    main()
