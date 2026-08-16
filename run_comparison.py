"""
Runs three experiment variants back to back and produces a single
comparison table + bar chart:

  1. centralized      -- no FL, no privacy, no blockchain (upper bound)
  2. fl_no_privacy     -- federated averaging only
  3. bcfl_hybrid        -- full pipeline: FL + blockchain ledger + DP + secure aggregation

This is the evidence piece for objectives 3 and 4: it shows what
utility (AUC/F1) costs you as you add privacy and decentralization.

Usage:
    python run_comparison.py --model lightweight --rounds 5 --clients 3
"""

import argparse
import os

import tensorflow as tf

import config
from data.dataset import CXR14Sequence, load_and_prepare_dataframe, split_dataframe
from models.model import build_global_model, build_lightweight_model
from blockchain.ledger import Blockchain
from privacy.mechanisms import apply_hybrid_privacy
from federated.client import train_client_fedavg
from federated.server import weighted_average
from evaluation.evaluate import evaluate_model, compare_experiments, save_results


def build_model(kind, num_classes):
    if kind == 'densenet':
        return build_global_model(num_classes=num_classes)
    return build_lightweight_model(num_classes=num_classes)


def run_centralized(train_df, val_df, unique_labels, model_kind, epochs=5):
    print("\n[1/3] Centralized baseline (no FL, no privacy) ...")
    train_gen = CXR14Sequence(train_df, config.IMAGES_DIR, unique_labels,
                               batch_size=config.BATCH_SIZE, shuffle=True,
                               preprocessing=True, augmentation=True)
    model = build_model(model_kind, len(unique_labels))
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=epochs, verbose=1)
    return model


def run_federated(train_df, unique_labels, model_kind, rounds, num_clients, use_privacy):
    label = "FL + BCFL hybrid privacy" if use_privacy else "FL only (no privacy)"
    print(f"\n[{'3/3' if use_privacy else '2/3'}] {label} ...")

    splits = split_dataframe(train_df, num_clients)
    client_seqs = [
        CXR14Sequence(s, config.IMAGES_DIR, unique_labels, batch_size=config.BATCH_SIZE,
                      shuffle=True, preprocessing=True, augmentation=True)
        for s in splits
    ]
    client_sizes = [len(c.df) for c in client_seqs]

    global_model = build_model(model_kind, len(unique_labels))
    global_weights = global_model.get_weights()
    ledger = Blockchain() if use_privacy else None

    for rnd in range(1, rounds + 1):
        print(f"  round {rnd}/{rounds}")
        raw_weights = []
        for seq in client_seqs:
            client_model = tf.keras.models.clone_model(global_model)
            w, _ = train_client_fedavg(client_model, seq, global_weights, epochs=config.LOCAL_EPOCHS)
            raw_weights.append(w)

        if use_privacy:
            protected = apply_hybrid_privacy(raw_weights)
            for i, w in enumerate(protected):
                ledger.add_block(f"client_{i}", Blockchain.hash_weights(w), rnd)
        else:
            protected = raw_weights

        global_weights = weighted_average(protected, client_sizes)

    global_model.set_weights(global_weights)

    if use_privacy:
        valid, msg = ledger.is_valid()
        print(f"  Blockchain check: {msg}")
        ledger.save(os.path.join(config.OUTPUT_DIR, 'comparison_bcfl_ledger.json'))

    return global_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['lightweight', 'densenet'], default='lightweight')
    p.add_argument('--rounds', type=int, default=config.NUM_ROUNDS)
    p.add_argument('--clients', type=int, default=config.NUM_CLIENTS)
    p.add_argument('--centralized-epochs', type=int, default=5)
    args = p.parse_args()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    train_df, val_df, test_df, unique_labels = load_and_prepare_dataframe()
    test_gen = CXR14Sequence(test_df, config.IMAGES_DIR, unique_labels, batch_size=config.BATCH_SIZE,
                              shuffle=False, preprocessing=True, augmentation=False)

    results = {}

    centralized_model = run_centralized(train_df, val_df, unique_labels, args.model,
                                         epochs=args.centralized_epochs)
    results['centralized'] = evaluate_model(centralized_model, test_gen, unique_labels)

    fl_model = run_federated(train_df, unique_labels, args.model, args.rounds, args.clients,
                              use_privacy=False)
    results['fl_no_privacy'] = evaluate_model(fl_model, test_gen, unique_labels)

    bcfl_model = run_federated(train_df, unique_labels, args.model, args.rounds, args.clients,
                                use_privacy=True)
    results['bcfl_hybrid'] = evaluate_model(bcfl_model, test_gen, unique_labels)

    print("\n=== Comparison ===")
    compare_experiments(results, save_path=os.path.join(config.OUTPUT_DIR, 'comparison_chart.png'))
    save_results(results, os.path.join(config.OUTPUT_DIR, 'comparison_results.json'))


if __name__ == '__main__':
    main()
