import sys
import argparse
import pickle
import tensorflow.keras as keras
import numpy as np
from cluster_intents import Sentence

def cos_loss(x, y):
    return keras.backend.abs(keras.losses.cosine_proximity(x, y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str)
    parser.add_argument('--output')
    parser.add_argument('--input')
    parser.add_argument('--encoded_dim', type=int)
    args = parser.parse_args()
    with open(args.input, 'rb') as inf:
        all_sents = pickle.load(inf)
    x_data = np.array([getattr(sent, args.feature) for sent in all_sents])
    x_data /= np.max(x_data)
    in_dim = x_data.shape[1]
    input_l = keras.layers.Input(shape=(in_dim,))
    encoded = keras.layers.Dense(args.encoded_dim,)(input_l)
    #encoded = keras.layers.Dense(args.encoded_dim, activation='relu', activity_regularizer=keras.regularizers.l1(10e-5))(input_l)
    reconstructed = keras.layers.Dense(in_dim, activation='sigmoid')(encoded)
    ae_model = keras.models.Model(input_l, reconstructed)
    encoder = keras.models.Model(input_l, encoded)

    if args.feature == 'boe':
        ae_model.compile(optimizer='adadelta', loss=cos_loss)
        print('cosine')
    else:
        ae_model.compile(optimizer='adadelta', loss='binary_crossentropy')
    ae_model.fit(x_data, x_data, epochs=200, shuffle=True)

    encoded = encoder.predict(x_data)
    for i, sent in enumerate(all_sents):
        setattr(sent, args.feature + '_enc', encoded[i])

    with open(args.output, 'wb') as outf:
        pickle.dump(all_sents, outf)
