"""The 11-class digit (7-segment) readout network, as a plain Python builder.

This is *exactly* the graph from ``03 - Train_CNN_Digital-Readout-Small-v2.ipynb``
(the model shipped as ``dig-class11_2000_s2``), lifted out of the notebook so a
training run can be reproduced and diffed from the command line.

Do not "improve" the topology here. The firmware runs TFLite for
Microcontrollers with a fixed op resolver, and the shipped models' op set
(ADD / CONV_2D / FULLY_CONNECTED / MAX_POOL_2D / MUL / RESHAPE / SOFTMAX) is the
compatibility contract. Any new layer type risks an ``AllocateTensors() failed``
on the device that looks like a corrupt file.

Contract:
  * input : float32 (N, 32, 20, 3), RAW 0-255 -- no normalisation. The first
            layer is BatchNormalization, which learns the scaling itself.
  * output: softmax over 11 classes; class 10 == NaN / blank ("N").
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

# Reach Keras through attribute access on `tf` rather than
# `from tensorflow.keras... import ...`. The submodule import form is fragile:
# `tensorflow.keras` is a lazily-loaded alias, not a real package directory, so
# anything occupying that name in site-packages breaks it. (A PyPI package
# literally called `tensorflow.keras` exists and is NOT published by the
# TensorFlow project.) Attribute access goes through TF's own lazy loader.
try:
    keras = tf.keras
except AttributeError:  # pragma: no cover - stand-alone Keras 3 install
    import keras  # type: ignore

__all__ = ["build_dig_class11", "PARAM_COUNT"]

# Parameter count of the notebook graph at the (32, 20, 3) input size. Asserted
# by the self-test below and printed by the training script, because a silent
# shape change (e.g. a transposed 20x32 input) still trains and still exports --
# it just does not match the device's ROI orientation.
PARAM_COUNT = 88023


def build_dig_class11(input_shape=(32, 20, 3)):
    """Build and compile the notebook-03 dig-class11 network.

    Layers, in order: Input -> BatchNormalization -> [Conv2D(32, 3x3, same,
    relu) -> MaxPool2D(2x2)] x3 -> Flatten -> Dense(256, relu) ->
    Dense(11, softmax).
    """
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.BatchNormalization()(inputs)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(11, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95),
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    m = build_dig_class11()
    m.summary()
    n = m.count_params()
    print(f"\ntotal params: {n}")
    if n != PARAM_COUNT:
        raise SystemExit(f"expected {PARAM_COUNT} params, got {n}")
    print("OK: parameter count matches the shipped dig-class11_2000_s2 graph")
