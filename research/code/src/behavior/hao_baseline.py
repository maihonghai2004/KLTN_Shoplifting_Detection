"""
Kiến trúc baseline LSTM-CNN Hybrid của Hảo (Keras).

Source gốc: KLTN.ipynb và test4.py của Hảo.
Input mặc định: (90, 56) — 90 frame × 56 đặc trưng hình học.

Tổng ~93K tham số (V1.0/V1.2). Đây là model baseline cho ablation study.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    BatchNormalization, Dense, Reshape, Input
)
from tensorflow.keras.optimizers import Adam


def build_hao_lstm_cnn(input_shape=(90, 56), num_classes=2, learning_rate=1e-4):
    """
    Build kiến trúc LSTM-CNN Hybrid của Hảo.

    Args:
        input_shape: (T, F) — số frame × số features (90, 56) hoặc (90, 34)
        num_classes: số lớp đầu ra (2 cho Normal/Shoplifting)
        learning_rate: lr cho Adam optimizer

    Returns:
        keras.Model đã compile.
    """
    T, F = input_shape
    # Số channel sau LSTM (hidden_size = 32)
    H = 32

    model = Sequential([
        Input(shape=input_shape),

        # Giai đoạn 1: LSTM học đặc trưng thời gian
        LSTM(H, return_sequences=True),
        LSTM(H, return_sequences=True),

        # Chuyển đổi sang không gian CNN
        Reshape((T, H, 1)),

        # Giai đoạn 2: Convolution học đặc trưng không gian
        Conv2D(64, kernel_size=(5, 5), strides=(2, 2),
               padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
               padding='same', activation='relu'),

        # Kết thúc: GAP + BN + Dense
        GlobalAveragePooling2D(),
        BatchNormalization(),

        Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def load_hao_checkpoint(checkpoint_path: str, input_shape=(90, 56)):
    """
    Load model Hảo từ file .h5.

    Hỗ trợ .h5 lưu bởi cả Keras 2 và Keras 3 (format khác nhau).
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    # Try direct load first
    try:
        return tf.keras.models.load_model(checkpoint_path, compile=False)
    except Exception as e:
        print(f"[Hao] Direct load failed: {e}")
        print("[Hao] Fallback: rebuild + load weights from h5")
        return _load_hao_from_keras3_h5(checkpoint_path, input_shape)


def _load_hao_from_keras3_h5(path, input_shape=(90, 56)):
    """Load weights từ .h5 file Keras 3 vào model Keras 2 (TF 2.15).

    File .h5 Keras 3 lưu weights theo tên layer, cần mapping thủ công.
    """
    import h5py
    import numpy as np

    model = build_hao_lstm_cnn(input_shape=input_shape)
    # Build model
    model.predict(np.zeros((1, *input_shape), dtype=np.float32), verbose=0)

    with h5py.File(path, 'r') as f:
        mw = f['model_weights']

        # Mapping: Keras 3 h5 layer names → weight arrays in order
        weight_arrays = []

        # LSTM 1: kernel, recurrent_kernel, bias
        lstm1 = mw['lstm_2']['sequential_1']['lstm_2']['lstm_cell']
        weight_arrays.append(np.array(lstm1['kernel']))
        weight_arrays.append(np.array(lstm1['recurrent_kernel']))
        weight_arrays.append(np.array(lstm1['bias']))

        # LSTM 2
        lstm2 = mw['lstm_3']['sequential_1']['lstm_3']['lstm_cell']
        weight_arrays.append(np.array(lstm2['kernel']))
        weight_arrays.append(np.array(lstm2['recurrent_kernel']))
        weight_arrays.append(np.array(lstm2['bias']))

        # Conv2D 1
        conv1 = mw['conv2d_2']['sequential_1']['conv2d_2']
        weight_arrays.append(np.array(conv1['kernel']))
        weight_arrays.append(np.array(conv1['bias']))

        # Conv2D 2
        conv2 = mw['conv2d_3']['sequential_1']['conv2d_3']
        weight_arrays.append(np.array(conv2['kernel']))
        weight_arrays.append(np.array(conv2['bias']))

        # BatchNormalization
        bn = mw['batch_normalization_1']['sequential_1']['batch_normalization_1']
        weight_arrays.append(np.array(bn['gamma']))
        weight_arrays.append(np.array(bn['beta']))
        weight_arrays.append(np.array(bn['moving_mean']))
        weight_arrays.append(np.array(bn['moving_variance']))

        # Dense
        dense = mw['dense_1']['sequential_1']['dense_1']
        weight_arrays.append(np.array(dense['kernel']))
        weight_arrays.append(np.array(dense['bias']))

    # Assign weights
    non_opt = [w for w in model.weights if 'optimizer' not in w.name.lower()]
    if len(weight_arrays) != len(non_opt):
        raise ValueError(f"Weight count mismatch: h5 has {len(weight_arrays)}, model has {len(non_opt)}")

    for w_var, w_val in zip(non_opt, weight_arrays):
        w_var.assign(w_val)

    print(f"[Hao] Loaded {len(weight_arrays)} weight arrays from Keras 3 h5: {path}")
    return model
