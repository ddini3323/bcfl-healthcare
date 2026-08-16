"""
Model definitions.

build_global_model  -> DenseNet121 transfer-learning model (recommended)
build_lightweight_model -> small CNN, useful for quick local testing on CPU
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121

import config


def build_global_model(input_shape=config.INPUT_SHAPE, num_classes=None, fine_tune_last_n=0):
    """DenseNet121 backbone pretrained on ImageNet, replaced classification
    head sized to the number of chest X-ray finding labels.

    fine_tune_last_n: number of trailing backbone layers to unfreeze.
    Leave at 0 to only train the new head (fastest, least data-hungry --
    good default for a federated setting where each client sees little data).
    """
    if num_classes is None:
        raise ValueError("num_classes must be provided (== len(unique_labels))")

    base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False
    if fine_tune_last_n > 0:
        for layer in base.layers[-fine_tune_last_n:]:
            layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    return models.Model(inputs=base.input, outputs=outputs, name='densenet121_cxr14')


def build_lightweight_model(input_shape=config.INPUT_SHAPE, num_classes=None):
    """Small CNN with no pretrained weights -- much faster to train locally
    on a laptop CPU, useful to sanity-check the FL/blockchain/privacy
    pipeline before switching to the full DenseNet121 model."""
    if num_classes is None:
        raise ValueError("num_classes must be provided (== len(unique_labels))")

    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid'),
    ], name='lightweight_cxr14')
