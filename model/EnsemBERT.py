import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Reshape, Dense, Concatenate, MultiHeadAttention, Softmax, Add, Multiply, Dot
)
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses



class ReduceSumLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)



class Attention(tf.keras.layers.Layer):
    def call(self, inputs):
        q, k, v = inputs
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output


# weighted sum layer
class WeightedSum(tf.keras.layers.Layer):
    def call(self, inputs):
        predictions, similarities = inputs
        return 0.5 * predictions + 0.5 * similarities


class EnsemBERTModel:
    def __init__(self, embedding_dim, max_posts, dense_units, num_heads, learning_rate):
        self.embedding_dim = embedding_dim
        self.max_posts = max_posts
        self.dense_units = dense_units
        self.num_heads = num_heads
        self.learning_rate = learning_rate
        self.model = None

    def build(self, out_weight, soft_weight, meta_weight):
    
        Input_Reddit = Input((self.max_posts, self.embedding_dim))
        Input_Description = Input((self.embedding_dim,))
        Input_Desc = Reshape((1, self.embedding_dim))(Input_Description)

        Att_Reddit = Attention()([Input_Reddit, Input_Reddit, Input_Reddit])
        Pooling_Posts = ReduceSumLayer(axis=1)(Att_Reddit)

        cross_attention_layer = MultiHeadAttention(key_dim=self.embedding_dim, num_heads=self.num_heads)
        cross_attention, cross_weights = cross_attention_layer(
            query=Input_Desc, key=Input_Reddit, value=Input_Reddit, return_attention_scores=True
        )
        cross_attention = Reshape((self.embedding_dim,))(cross_attention)

        # similarity calculation
        Inputs_Choices = []
        Output_Similarity = []

        for i in range(4):  number of choices
            inp_choice_i = Input((self.embedding_dim,))
            Inputs_Choices.append(inp_choice_i)
            Cosine_Sim = Dot(axes=-1, normalize=True)([cross_attention, inp_choice_i])
            Output_Similarity.append(Cosine_Sim)

        Concatenated_Similarities = Concatenate()(Output_Similarity)
        Softmax_Similarities = Softmax(name='soft')(Concatenated_Similarities)

        Weighted_Sum = Add()([
            Multiply()([inp_choice, tf.expand_dims(Softmax_Similarities[:, i], axis=-1)])
            for i, inp_choice in enumerate(Inputs_Choices)
        ])

        #meta
        Merge_Layer = Add()([Pooling_Posts, Weighted_Sum])
        dense_layer = Dense(self.dense_units, activation='relu')(Merge_Layer)
        out_soft_1 = Dense(4, activation='softmax', name='out')(dense_layer)

        predictions_concatenate = WeightedSum()([out_soft_1, Softmax_Similarities])
        meta_predictions = Dense(4, activation='softmax', name='meta')(predictions_concatenate)

        #model identification
        self.model = Model(
            inputs=[Input_Reddit, Input_Description] + Inputs_Choices,
            outputs=[out_soft_1, Softmax_Similarities, meta_predictions]
        )

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss={
                'out': losses.SparseCategoricalCrossentropy(),
                'soft': losses.SparseCategoricalCrossentropy(),
                'meta': losses.SparseCategoricalCrossentropy()
            },
            loss_weights={'out': out_weight, 'soft': soft_weight, 'meta': meta_weight}
        )

    def summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet. Call 'build()' first.")

    def train(self, *args, **kwargs):
        if self.model:
            return self.model.fit(*args, **kwargs)
        else:
            print("Model not built yet. Call 'build()' first.")

    def predict(self, *args, **kwargs):
        if self.model:
            return self.model.predict(*args, **kwargs)
        else:
            print("Model not built yet. Call 'build()' first.")


