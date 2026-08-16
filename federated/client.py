"""
Client-side local training. Supports plain local SGD (used for FedAvg)
and a FedProx proximal term to handle non-IID client data, which is the
"novel" component relative to plain FedAvg (objective 1).
"""

import tensorflow as tf

import config


def train_client_fedavg(model, train_seq, global_weights, epochs=config.LOCAL_EPOCHS,
                         lr=config.LEARNING_RATE):
    """Standard local training: just fit on local data starting from the
    current global weights."""
    model.set_weights(global_weights)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_seq, epochs=epochs, verbose=0)
    return model.get_weights(), history.history


def train_client_fedprox(model, train_seq, global_weights, mu=config.FEDPROX_MU,
                          epochs=config.LOCAL_EPOCHS, lr=config.LEARNING_RATE):
    """FedProx local training: adds a proximal term that penalizes local
    weights drifting far from the global model. This limits client drift
    when client data distributions differ (non-IID), which is exactly the
    situation you'd expect across different hospitals. Set mu=0 to recover
    plain FedAvg local training.
    """
    model.set_weights(global_weights)
    global_weights_tensors = [tf.constant(w) for w in global_weights]
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    bce = tf.keras.losses.BinaryCrossentropy()

    epoch_losses = []
    for _ in range(epochs):
        batch_losses = []
        for i in range(len(train_seq)):
            x_batch, y_batch = train_seq[i]
            if len(x_batch) == 0:
                continue
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = bce(y_batch, preds)
                if mu > 0:
                    prox_term = tf.add_n([
                        tf.reduce_sum(tf.square(w - gw))
                        for w, gw in zip(model.trainable_weights, global_weights_tensors)
                    ])
                    loss = loss + (mu / 2.0) * prox_term
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            batch_losses.append(float(loss.numpy()))
        if batch_losses:
            epoch_losses.append(sum(batch_losses) / len(batch_losses))
        train_seq.on_epoch_end()

    return model.get_weights(), {'loss': epoch_losses}
