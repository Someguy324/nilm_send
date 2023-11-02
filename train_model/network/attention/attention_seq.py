import tensorflow as tf

from train_model.network.attention.attention_layer.attention_layer import AttentionLayer


def create_attention_concat_seq(window_size):
    attention, attention_weight = create_attention(window_size)
    mlp = create_mlp(window_size)
    combined_input = tf.keras.layers.concatenate([attention.output[0], mlp.output])
    final_layer = tf.keras.layers.Dense(128, activation='relu')(combined_input)
    output_layer = tf.keras.layers.Dense(window_size, activation='linear', name='output')(final_layer)
    model = tf.keras.Model(inputs=[attention.input, mlp.input], outputs=[output_layer, attention.output[1]])
    return model


def create_attention(window_size):
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

    output = tf.keras.layers.Multiply(name="output_layer")([regression_output, classification_output])

    full_model = tf.keras.Model(inputs=input_data, outputs=[output, classification_output], name="LDwA")
    attention_model = tf.keras.Model(inputs=input_data, outputs=weights)

    return full_model, attention_model


def create_mlp(input_window_length):
    input_layer = tf.keras.layers.Input(shape=(input_window_length, 3))
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    second_layer = tf.keras.layers.Dense(32, activation='elu')(flatten_layer)
    third_layer = tf.keras.layers.Dense(16, activation='elu')(second_layer)
    model = tf.keras.Model(inputs=input_layer, outputs=third_layer)
    return model
