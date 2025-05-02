from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling1D, Concatenate, Lambda

# === Transformer Block ===
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, mask=None, training=False):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def build(self, input_shape):
        # Explicitly define build method to address warnings
        self.built = True
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout": self.dropout_rate
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# === Attention Pooling ===
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense = Dense(1)

    def call(self, x, mask=None):
        scores = self.dense(x)  # shape: (batch, seq, 1)
        scores = tf.squeeze(scores, axis=-1)  # shape: (batch, seq)
        # mask out the scores of the padded tokens
        if mask is not None:
            scores += tf.cast(~mask, tf.float32) * -1e9
        weights = tf.nn.softmax(scores, axis=1)
        weighted_sum = tf.reduce_sum(x * weights[..., tf.newaxis], axis=1)
        return weighted_sum
    
    def build(self, input_shape):
        # Explicitly define build method to address warnings
        self.built = True
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim
        })
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# === Cross Attention helper ===
def cross_attention(query, key_value, mask, num_heads=2, key_dim=64):
    return layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        query[:, tf.newaxis, :], key_value, attention_mask=mask
    )[:, 0, :]


# === Lambda layer output shape functions ===
def expand_mask_output_shape(input_shape):
    """Output shape function for the mask expansion Lambda layer"""
    return (input_shape[0], 1, 1, input_shape[1])


# === Model ===
def build_classifier(max_particles=25, max_sv=4, global_features=11, particle_features=20, sv_features=6):
    # --- Inputs ---
    global_input = Input(shape=(global_features,), name="global_input")
    particle_input = Input(shape=(max_particles, particle_features), name="particle_input")
    particle_mask_input = Input(shape=(max_particles,), dtype=tf.bool, name="particle_mask")
    sv_input = Input(shape=(max_sv, sv_features), name="sv_input")
    sv_mask_input = Input(shape=(max_sv,), dtype=tf.bool, name="sv_mask")

    # --- Global Branch ---
    global_x = Dense(128, activation="relu")(global_input)
    global_x = Dropout(0.3)(global_x)
    global_x = Dense(128, activation="relu")(global_x)
    global_x = Dropout(0.3)(global_x)
    global_x = Dense(64, activation="relu")(global_x)
    global_x = Dropout(0.3)(global_x)

    # --- Particle Branch ---
    particle_embed = Dense(64)(particle_input)
    # Specify output_shape for Lambda layer
    particle_mask_expanded = Lambda(
        lambda x: tf.cast(x[:, tf.newaxis, tf.newaxis, :], tf.float32),
        output_shape=expand_mask_output_shape
    )(particle_mask_input)

    particle_x = particle_embed
    for _ in range(3):
        particle_x = TransformerEncoder(64, num_heads=4, ff_dim=128)(particle_x, mask=particle_mask_expanded)

    particle_attn = AttentionPooling(64)(particle_x, particle_mask_input)
    particle_attn = Dropout(0.3)(particle_attn)
    
    # --- SV Branch ---
    sv_embed = Dense(32)(sv_input)
    # Specify output_shape for Lambda layer
    sv_mask_expanded = Lambda(
        lambda x: tf.cast(x[:, tf.newaxis, tf.newaxis, :], tf.float32),
        output_shape=expand_mask_output_shape
    )(sv_mask_input)

    sv_x = sv_embed
    for _ in range(3):
        sv_x = TransformerEncoder(32, num_heads=2, ff_dim=64)(sv_x, mask=sv_mask_expanded)

    sv_attn = AttentionPooling(32)(sv_x, sv_mask_input)
    sv_attn = Dropout(0.3)(sv_attn)
    
    # --- Global Cross Attention ---
    global_cross_particle = cross_attention(global_x, particle_embed, particle_mask_expanded, num_heads=2, key_dim=64)
    global_cross_particle = Dropout(0.3)(global_cross_particle)

    global_cross_sv = cross_attention(global_x, sv_embed, sv_mask_expanded, num_heads=1, key_dim=32)
    global_cross_sv = Dropout(0.3)(global_cross_sv)

    # --- Fusion ---
    x = Concatenate()([global_x, particle_attn, sv_attn, global_cross_particle, global_cross_sv])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(3, activation="softmax")(x)

    return Model(
        inputs=[global_input, particle_input, particle_mask_input, sv_input, sv_mask_input],
        outputs=output
    )