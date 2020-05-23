import tensorflow as tf
import json
import pandas as pd
import numpy as np
import codecs
import csv
import os
import streamlit as st
import tensorflow_datasets as tfds
from bert_serving.client import BertClient
import time
from tqdm import tqdm
import tensorboard
import datetime
import nltk
import re

TRAIN_DATA_FILE ='train.csv'#'train_micro.csv'
TRAIN_DATA_FILE_MINI = 'trainMini.csv'
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 64000

d_model = 512
num_layers = 6
dff = 2048
num_heads = 16
EPOCHS = 50
MAX_LENGTH = 40



def get_angles2(pos, i,le, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    a = le - pos


    return a * angle_rates

def positional_encoding2(position,le, d_model):
    
    lens = np.zeros((le,position,d_model))

    for li in range(le):
        angle_rads = get_angles2(np.arange(position)[:, np.newaxis],li,
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        # print (type(angle_rads))
        lens[li,:,:] = angle_rads
    
    pos_encoding = lens[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)




def remove(list):
    pattern = '[0-9]+'
    list = [re.sub(pattern, '#', i) for i in list]
    return list





def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    # print("AHHHHHHHHHHHHHHHHHHHHHHHHHHH",position)
    # print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",length)
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                        np.arange(d_model)[np.newaxis, :],
                        d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(
    q, k, v, None)
    print ('Attention weights are:')
    print (temp_attn)
    print ('Output is:')
    print (temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights






def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
    tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
    tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding2(41,41,
                                                self.d_model)
        # self.maximum_position_encoding = maximum_position_encoding

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                        for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask, seq_len ,length = 20):
        # seq_len = tf.shape(x)[1]
        
        # with tf.Session() as sess:
        # print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",seq_len.numpy())

        # if training:
            # self.pos_encoding = positional_encoding(self.maximum_position_encoding,40,
                                                # self.d_model)
        # else:
            # self.pos_encoding = positional_encoding(self.maximum_position_encoding,length,
                                                # self.d_model)

        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if training:
            x += self.pos_encoding[:,seq_len,:seq_len, :]
        else:
            # print('X shape encoder: ',x.shape)
            # print("ecoding shape encoder : ",self.pos_encoding[:, length,: seq_len  , :].shape)
            x += self.pos_encoding[:, length,:seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding,rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding2(41,41,d_model)

        # self.maximum_position_encoding =maximum_position_encoding

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                        for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask,seq_len, length =20):
        # seq_len = tf.shape(x)[1]

        # with tf.Session() as sess:
        #     print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH",seq_len)


        attention_weights = {}

        # seq_len = tf.shape(x)[1]

        # if training:
            # self.pos_encoding = positional_encoding(self.maximum_position_encoding,40,
                                                # self.d_model)
        # else:
            # self.pos_encoding = positional_encoding(self.maximum_position_encoding,length,
                                                # self.d_model)

        # x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if training:
            x += self.pos_encoding[:,seq_len,:seq_len, :]
        else:
            # print('X shape decoder : ',x.shape)
            # print("ecoding shape decoder: ",self.pos_encoding[:, length,:seq_len , :].shape)
            x += self.pos_encoding[:, length,:seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target,rate=0.1):
        super(Transformer, self).__init__()

        print('TArget',target_vocab_size )
        self. embeding_layer = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # self.Wl = tf.keras.layers.Embedding(20, 1)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                            input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                            target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask,length =20):

        embedding_out_inp = self.embeding_layer(inp)
        embedding_out_tar = self.embeding_layer(tar)
        seq_lenE = tf.shape(inp)[1]
        seq_lenD = tf.shape(tar)[1]

        # print("AHHHHHHHHHHHHHHHHHHH",length)
        # lemembd = self.Wl(tar)
        # embedding_out_tar =  tf.bitwise.bitwise_xor(embedding_out_inp, lemembd, name=None)

        enc_output = self.encoder(embedding_out_inp, training, enc_padding_mask,seq_lenE,length )  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            embedding_out_tar, enc_output, training, look_ahead_mask, dec_padding_mask,seq_lenD ,length )

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def load_dataset():
    df = pd.read_csv(TRAIN_DATA_FILE,usecols=['title', 'text'],encoding='utf-8' )
    # df = pd.read_csv(TRAIN_DATA_FILE_MINI,usecols=['title', 'text'],encoding='utf-8' )
    # data_1,NB_WORDS,tokenizer = preprocessing(TRAIN_DATA_FILE,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS,df)

    texts_aux = [nltk.tokenize.sent_tokenize(doc,language='spanish') for doc in tqdm(df.text)]

    train_examples = remove([sentence for doc in texts_aux for sentence in doc])
    return train_examples


train_examples = load_dataset()[:2000000]

val_examples= train_examples[1980000:] 
train_examples = tf.data.Dataset.from_tensor_slices((train_examples[:1980000],train_examples[:1980000]))#)
val_examples = tf.data.Dataset.from_tensor_slices((val_examples,val_examples))

try:
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file('tokenizer')
    # 'Lo cargue'
except:
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (sentence.numpy() for sentence,sentencesI in train_examples), target_vocab_size=2**14)
    tokenizer.save_to_file('tokenizer')



input_vocab_size =  tokenizer.vocab_size + 2
target_vocab_size = input_vocab_size
dropout_rate = 0.1

BUFFER_SIZE = 20000
BATCH_SIZE = 64


def encode(sent,sent1):
    sent = [tokenizer.vocab_size] + tokenizer.encode(
    sent.numpy()) + [tokenizer.vocab_size+1]

    return sent ,sent

# test = encode("El perro de mi casa ladra ","El perro de mi casa ladra ")

# test
def tf_encode(sent,sent1):
    # result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    result , resul1= tf.py_function(encode, [sent,sent], [tf.int64,tf.int64])

    # st.write(len(result))
    result.set_shape([None])
    resul1.set_shape([None])
    # st.write(result[0])

    return result,resul1

def filter_max_length(x,y ,max_length=MAX_LENGTH):

    return tf.logical_and(tf.size(x) <= max_length,
                            tf.size(y) <= max_length)


train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE,padded_shapes=([None],[None]),drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE,padded_shapes=([None],[None]),drop_remainder=True)



learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


################LA CONCRETA#################
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                    True,
                                    enc_padding_mask,
                                    combined_mask,
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)



current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (inp,tar)) in enumerate(train_dataset):
            train_step(inp, tar)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)  
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            # tf.summary.flush()
            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

            # if batch % 5000 == 0:
            #     ckpt_save_path = ckpt_manager.save()
            #     print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
            #                                                 ckpt_save_path))
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))

        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                train_loss.result(),
                                                train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))








def evaluate(inp_sentence,length):
    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
    # print("ajjjjqwod",len(inp_sentence))

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                    output,
                                                    False,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask,length=length)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def reconstruct(sentence,length ,plot=''):
    result, attention_weights = evaluate(sentence,length)

    predicted_sentence = tokenizer.decode([i for i in result
                                            if i < tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    # if plot:
        # plot_attention_weights(attention_weights, sentence, result, plot)


reconstruct('la carrera por ser el candidato demócrata contra Trump es ahora una lucha entre dos, Joe Biden moderado y Bernie Sanders del ala izquierda.',40)
