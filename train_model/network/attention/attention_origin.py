import tensorflow as tf

from train_model.network.attention.attention_layer.attention_layer import AttentionLayer


def create_attention_origin(window_size):
    input_data = tf.keras.Input(shape=(window_size, 1))

    # CLASSIFICATION SUBNETWORK
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding='same', activation='elu')(input_data)
    x = tf.keras.layers.Conv1D(filters=30, kernel_size=8, padding='same', activation='elu')(x)
    x = tf.keras.layers.Conv1D(filters=40, kernel_size=6, padding='same', activation='elu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, padding='same', activation='elu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, padding='same', activation='elu')(x)
    x = tf.keras.layers.Conv1D(filters=50, kernel_size=5, padding='same', activation='elu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=1024, activation='elu', kernel_initializer='he_normal')(x)
    classification_output = tf.keras.layers.Dense(units=window_size, activation='sigmoid',
                                                  name="classification_output")(x)

    # REGRESSION SUBNETWORK
    y = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='elu')(input_data)
    y = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='elu')(y)
    y = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='elu')(y)
    y = tf.keras.layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='elu')(y)
    y = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="tanh", return_sequences=True),
                                      merge_mode="concat")(y)
    y, weights = AttentionLayer(units=128)(y)
    y = tf.keras.layers.Dense(128, activation='elu')(y)
    regression_output = tf.keras.layers.Dense(window_size, activation='elu', name="regression_output")(y)

    output = tf.keras.layers.Multiply(name="output")([regression_output, classification_output])

    full_model = tf.keras.Model(inputs=input_data, outputs=[output, classification_output], name="LDwA")
    attention_model = tf.keras.Model(inputs=input_data, outputs=weights)

    return full_model, attention_model
