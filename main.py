"""
End-to-end BCFL Healthcare pipeline.

Run with:
    python main.py --model lightweight --rounds 5 --clients 3
    python main.py --model densenet --rounds 10 --clients 5 --fedprox

See config.py to change dataset paths and hyperparameters.
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf

import config
from data.dataset import CXR14Sequence, load_and_prepare_dataframe, split_dataframe
from models.model import build_global_model, build_lightweight_model
from blockchain.ledger import Blockchain
from privacy.mechanisms import apply_hybrid_privacy, estimate_epsilon
from federated.client import train_client_fedavg, train_client_fedprox
from federated.server import weighted_average
from evaluation.evaluate import evaluate_model, save_results


def parse_args():
    p = argparse.ArgumentParser(description="Blockchain-enabled Federated Learning for Chest X-ray classification")
    p.add_argument('--model', choices=['lightweight', 'densenet'], default='lightweight',
                    help="lightweight = fast CNN for local CPU testing; densenet = DenseNet121 transfer learning")
    p.add_argument('--rounds', type=int, default=config.NUM_ROUNDS)
    p.add_argument('--clients', type=int, default=config.NUM_CLIENTS)
    p.add_argument('--epochs', type=int, default=config.LOCAL_EPOCHS)
    p.add_argument('--fedprox', action='store_true', help="use FedProx local training instead of plain FedAvg")
    p.add_argument('--no-privacy', action='store_true', help="disable DP + secure aggregation (baseline FL run)")
    p.add_argument('--tag', default='bcfl_hybrid', help="name for this run, used in output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("Loading and splitting dataset ...")
    train_df, val_df, test_df, unique_labels = load_and_prepare_dataframe()
    num_classes = len(unique_labels)
    print(f"  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}  classes={num_classes}")

    train_splits = split_dataframe(train_df, args.clients)
    train_clients = [
        CXR14Sequence(split, config.IMAGES_DIR, unique_labels, batch_size=config.BATCH_SIZE,
                      shuffle=True, preprocessing=True, augmentation=False)
        for split in train_splits
    ]
    val_gen = CXR14Sequence(val_df, config.IMAGES_DIR, unique_labels, batch_size=config.BATCH_SIZE,
                             shuffle=False, preprocessing=True, augmentation=False)
    test_gen = CXR14Sequence(test_df, config.IMAGES_DIR, unique_labels, batch_size=config.BATCH_SIZE,
                              shuffle=False, preprocessing=True, augmentation=False)

    print(f"Building global model ({args.model}) ...")
    if args.model == 'densenet':
        global_model = build_global_model(num_classes=num_classes)
    else:
        global_model = build_lightweight_model(num_classes=num_classes)
    global_weights = global_model.get_weights()

    ledger = Blockchain()
    client_sizes = [len(c.df) for c in train_clients]
    history = {'round': [], 'val_loss': [], 'val_accuracy': []}

    val_model = tf.keras.models.clone_model(global_model)
    val_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for rnd in range(1, args.rounds + 1):
        print(f"\n=== Federated Round {rnd}/{args.rounds} ===")
        raw_client_weights = []

        for idx, client_seq in enumerate(train_clients):
            client_name = f"client_{idx}"
            print(f"  Training {client_name} ({len(client_seq.df)} samples) ...")

            client_model = tf.keras.models.clone_model(global_model)
            if args.fedprox:
                weights, hist = train_client_fedprox(client_model, client_seq, global_weights,
                                                       epochs=args.epochs)
            else:
                weights, hist = train_client_fedavg(client_model, client_seq, global_weights,
                                                      epochs=args.epochs)
            raw_client_weights.append(weights)
            print(f"    local history: {hist}")

        # --- Privacy (objective 3): clip + DP noise, optionally + secure aggregation masks
        if args.no_privacy:
            protected_weights = raw_client_weights
        else:
            protected_weights = apply_hybrid_privacy(raw_client_weights)

        # --- Blockchain (objective 2): log a hash of each client's protected update
        for idx, w in enumerate(protected_weights):
            model_hash = Blockchain.hash_weights(w)
            ledger.add_block(f"client_{idx}", model_hash, rnd)

        # --- Aggregation (objective 1)
        global_weights = weighted_average(protected_weights, client_sizes)
        global_model.set_weights(global_weights)

        # --- Round validation
        val_model.set_weights(global_weights)
        val_loss, val_acc = val_model.evaluate(val_gen, verbose=0)
        print(f"  Round {rnd}: val_loss={val_loss:.4f}  val_accuracy={val_acc:.4f}")
        history['round'].append(rnd)
        history['val_loss'].append(float(val_loss))
        history['val_accuracy'].append(float(val_acc))

    # --- Blockchain integrity check
    valid, msg = ledger.is_valid()
    print(f"\nBlockchain integrity check: {msg}")
    ledger.save(config.BLOCKCHAIN_LOG)
    print(f"Ledger saved to {config.BLOCKCHAIN_LOG} ({len(ledger.chain)} blocks)")

    # --- Privacy budget estimate
    if not args.no_privacy:
        eps = estimate_epsilon(args.rounds, args.clients)
        print(f"Estimated DP budget: epsilon ~= {eps:.2f} at delta={config.DP_DELTA} "
              "(coarse composition bound -- use tensorflow-privacy/opacus for a tight accountant)")

    # --- Final test-set evaluation (objective 4)
    print("\nEvaluating final global model on held-out test set ...")
    global_model.set_weights(global_weights)
    results = evaluate_model(global_model, test_gen, unique_labels)
    print(f"Mean AUC-ROC: {results['mean_auc']}")
    print(f"Mean F1:      {results['mean_f1']}")

    metrics_path = os.path.join(config.OUTPUT_DIR, f"{args.tag}_metrics.json")
    save_results({'round_history': history, 'test_results': results}, metrics_path)
    print(f"Saved metrics to {metrics_path}")

    global_model.save(os.path.join(config.CHECKPOINT_DIR, f"{args.tag}_final_model.keras"))
    print("Done.")


if __name__ == '__main__':
    main()
