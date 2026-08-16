"""
Minimal but genuine blockchain used to log every client model update.

Each block references the previous block's hash, so any change to a past
block (client name, model hash, round, timestamp) breaks the chain and is
detectable via Blockchain.is_valid(). This is a local, single-node chain
(no consensus/mining) -- appropriate for a federated-learning integrity
log, not a claim of a distributed public blockchain.
"""

import hashlib
import json
import time


class Block:
    def __init__(self, index, client_name, model_hash, prev_hash, round_num, timestamp=None):
        self.index = index
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.client_name = client_name
        self.model_hash = model_hash
        self.round = round_num
        self.prev_hash = prev_hash
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'client': self.client_name,
            'model_hash': self.model_hash,
            'round': self.round,
            'prev_hash': self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'client': self.client_name,
            'model_hash': self.model_hash,
            'round': self.round,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
        }


class Blockchain:
    def __init__(self):
        self.chain = [self._genesis_block()]

    @staticmethod
    def _genesis_block():
        return Block(0, 'genesis', '0', '0', 0, timestamp=0)

    def add_block(self, client_name, model_hash, round_num):
        prev_block = self.chain[-1]
        new_block = Block(len(self.chain), client_name, model_hash, prev_block.hash, round_num)
        self.chain.append(new_block)
        return new_block

    def is_valid(self):
        """Returns (bool, message). Walks the chain checking that each
        block's stored hash matches its recomputed hash, and that each
        block correctly points at the previous block's hash."""
        for i in range(1, len(self.chain)):
            curr, prev = self.chain[i], self.chain[i - 1]
            if curr.hash != curr.compute_hash():
                return False, f"Block {i} content does not match its hash (tampered)."
            if curr.prev_hash != prev.hash:
                return False, f"Block {i} does not correctly reference block {i-1} (broken chain)."
        return True, "Chain is valid."

    def show_ledger(self):
        for block in self.chain:
            print(block.to_dict())

    def save(self, path):
        with open(path, 'w') as f:
            json.dump([b.to_dict() for b in self.chain], f, indent=2)

    @staticmethod
    def hash_weights(weights):
        """Deterministic SHA256 hash of a list of numpy weight arrays."""
        hasher = hashlib.sha256()
        for w in weights:
            hasher.update(w.tobytes())
        return hasher.hexdigest()
