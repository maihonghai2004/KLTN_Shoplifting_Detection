"""
Loader cho mô hình V1.4 (BiLSTM-CNN, Keras 3 .keras) chạy trên TF 2.15 (Keras 2).
(Chuyển từ KLTN/src/pipeline.py sang để backend chạy độc lập.)
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def _focal_loss_dummy(alpha=0.25, gamma=2.0):
    def f(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        oh = tf.cast(tf.one_hot(y_true, depth=tf.shape(y_pred)[-1]), y_pred.dtype)
        p = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
        ce = -oh * tf.math.log(p)
        return tf.reduce_mean(tf.reduce_sum(alpha * tf.math.pow(1 - p, gamma) * ce, axis=-1))
    return f


def _build_v14_bilstm(input_shape=(90, 56), num_classes=2):
    from tensorflow.keras.layers import (
        Bidirectional, Dropout as KDropout, Reshape,
        Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dense,
    )
    T, _F = input_shape
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)), KDropout(0.3),
        Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)), KDropout(0.3),
        Reshape((T, 64, 1)),
        Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='relu'),
        MaxPooling2D((2, 2), strides=(2, 2)), KDropout(0.3),
        Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
        GlobalAveragePooling2D(), BatchNormalization(), KDropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])


def _load_v14_from_keras3_zip(path, input_shape=(90, 56)):
    """Đọc weights từ file .keras (Keras 3 zip) gán vào model Keras 2."""
    import h5py

    model = _build_v14_bilstm(input_shape)
    model.predict(np.zeros((1, *input_shape), dtype=np.float32), verbose=0)

    with zipfile.ZipFile(path, 'r') as z:
        h5_data = z.read('model.weights.h5')

    with h5py.File(io.BytesIO(h5_data), 'r') as f:
        weight_map = [
            ('layers/bidirectional/forward_layer/cell/vars/0',
             'layers/bidirectional/forward_layer/cell/vars/1',
             'layers/bidirectional/forward_layer/cell/vars/2'),
            ('layers/bidirectional/backward_layer/cell/vars/0',
             'layers/bidirectional/backward_layer/cell/vars/1',
             'layers/bidirectional/backward_layer/cell/vars/2'),
            ('layers/bidirectional_1/forward_layer/cell/vars/0',
             'layers/bidirectional_1/forward_layer/cell/vars/1',
             'layers/bidirectional_1/forward_layer/cell/vars/2'),
            ('layers/bidirectional_1/backward_layer/cell/vars/0',
             'layers/bidirectional_1/backward_layer/cell/vars/1',
             'layers/bidirectional_1/backward_layer/cell/vars/2'),
            ('layers/conv2d/vars/0', 'layers/conv2d/vars/1'),
            ('layers/conv2d_1/vars/0', 'layers/conv2d_1/vars/1'),
            ('layers/batch_normalization/vars/0', 'layers/batch_normalization/vars/1',
             'layers/batch_normalization/vars/2', 'layers/batch_normalization/vars/3'),
            ('layers/dense/vars/0', 'layers/dense/vars/1'),
        ]
        all_weights = []
        for group in weight_map:
            for p in group:
                all_weights.append(np.array(f[p]))
        non_opt = [w for w in model.weights if 'optimizer' not in w.name.lower()]
        if len(all_weights) != len(non_opt):
            raise ValueError(f"Weight mismatch: zip {len(all_weights)} vs model {len(non_opt)}")
        for w_var, w_val in zip(non_opt, all_weights):
            w_var.assign(w_val)
    return model


def load_v14(path, input_shape=(90, 56)):
    """Thử load trực tiếp -> zip loader -> rebuild+load_weights."""
    try:
        return load_model(path, custom_objects={'f': _focal_loss_dummy(),
                                                 'loss_fn': _focal_loss_dummy()}, compile=False)
    except Exception:
        try:
            return _load_v14_from_keras3_zip(path, input_shape)
        except Exception:
            m = _build_v14_bilstm(input_shape)
            m.load_weights(path)
            return m
