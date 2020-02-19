import logging
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, CuDNNLSTM, Bidirectional, CuDNNLSTM, ELU, Dropout, LeakyReLU, Conv1D, BatchNormalization
from keras.optimizers import Adam

import librosa
import numpy as np

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from callbacks import CustomModelCheckpoint, CustomTensorBoard

from generator import BatchGenerator
from attention_layer import Attention 

phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']


def create_callbacks(saved_weights_name, tensorboard_logs, model_to_save):
    # makedirs(tensorboard_logs)

    early_stop = EarlyStopping(
        monitor='loss',
        min_delta=0.01,
        patience=5,
        mode='min',
        verbose=1
    )
    checkpoint = CustomModelCheckpoint(
        model_to_save=model_to_save,
        filepath=saved_weights_name,  # + '{epoch:02d}.h5',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        period=1
    )
    reduce_on_plateau = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=2,
        verbose=1,
        mode='min',
        epsilon=0.01,
        cooldown=0,
        min_lr=0
    )
    tensorboard = CustomTensorBoard(
        log_dir=tensorboard_logs,
        write_graph=True,
        write_images=True,
    )
    return [early_stop, checkpoint, reduce_on_plateau, tensorboard]


def load_vocab():
    phn2idx = {phn: idx for idx, phn in enumerate(phns)}
    idx2phn = {idx: phn for idx, phn in enumerate(phns)}

    return phn2idx, idx2phn

# returns list of wav_file_path and phn_file_path tuples [("<wav_file_path>", "<phn_file_path>"), ...]
def create_wav_phn_file_path_pairs(wav_dir, phn_dir):
    ret_list = []
    wavs = [wav for wav in os.listdir(wav_dir) if wav.lower().endswith(".wav")]

    for wav_file in wavs:
        phn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
        
        wav_file_path = os.path.join(wav_dir, wav_file)
        phn_file_path = os.path.join(phn_dir, phn_file)

        ret_list.append((wav_file_path, phn_file_path))

    return ret_list

# sr: sample rate
# time_window: determines win_length of each frame of audio (in seconds)
def generate_mel_phns_target_pair():
    train_wav_dir = "./train/wav"
    train_phn_dir = "./train/phn"
    train_wavs = [wav for wav in os.listdir(
        train_wav_dir) if wav.lower().endswith(".wav")]
    train_phns = [phn for phn in os.listdir(
        train_phn_dir) if phn.lower().endswith(".phn")]

    
    wav_base_path = train_wav_dir
    phn_base_path = train_phn_dir
    wav_files = train_wavs
    phn_files = train_phns
    sr=16000
    time_window=0.025
    num_phns = len(['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
        'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
        'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
        'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
        'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'])
    batch_size = 32
    while True:
        # # not sure if randomizing order matters, but just doing it to be safe
        # random.shuffle(wav_files)
        # random.shuffle(phn_files)
        for wav_file, phn_file in zip(wav_files, phn_files):
            print(os.path.join(wav_base_path, wav_file))

            # get amplitude graph
            x, sr = librosa.load(os.path.join(wav_base_path, wav_file), sr=sr)
            # print(x.shape)

            # get power-spectrogram (a power-spectrogram is just the amplitude squared)
            # n_fft: length of the windowed signal after padding with zeros. 512 is recommended for speech processing 
            # hop_length: number of audio samples between adjacent STFT columns.
            # win_length: Each frame of audio is windowed by window() of length win_length and then padded with zeros to match n_fft.
            n_fft = 512
            hop_length = 80
            win_length = int(sr * time_window)
            D = librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
            mag = np.abs(D)**2

            # compute the mel-spectrogram
            S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

            # convert from power-spectrogram to decibel units
            S_db = librosa.power_to_db(S)


            # 
            phn2idx, _ = load_vocab()
            num_timesteps = S_db.shape[1]
            phns = np.zeros(shape=(num_timesteps, num_phns))
            for line in open(os.path.join(phn_base_path, phn_file), 'r').read().splitlines():
                start_point, _, phn = line.split()
                bnd = int(start_point) // hop_length

                # one hot encoding of correct phoneme
                one_hot = np.zeros((num_phns))
                one_hot[phn2idx[phn]] = 1.0
                phns[bnd:] = one_hot

            # print("S_db", S_db.shape)
            # print("phns", phns.shape)

            duration = 2
            n_timesteps = (duration * sr) // hop_length + 1
            # print(n_timesteps)
            # Padding or crop
            S_db = librosa.util.fix_length(S_db, n_timesteps, axis=1)
            phns = librosa.util.fix_length(phns, n_timesteps, axis=0)

            S_db = S_db.transpose()

            # reshape to fit lstm 
            batch_size = 1
            S_db = np.reshape(S_db, (batch_size, S_db.shape[0], S_db.shape[1]))
            phns = np.reshape(phns, (batch_size, phns.shape[0], phns.shape[1]))

            print("S_db", S_db.shape)
            print("phns", phns.shape)

            # S_db.shape = (batch_size i.e. num_samples, num_time_steps per sample, num_features at each time_step)
            # phns.shape = (batch_size i.e. num_samples, num_time_steps per sample, num_phns)
            yield S_db, phns


# sr: sample rate
# time_window: determines win_length of each frame of audio (in seconds)
def extract_mel_phns_target_pairs():
    train_wav_dir = "./train/wav"
    train_phn_dir = "./train/phn"
    train_wavs = [wav for wav in os.listdir(
        train_wav_dir) if wav.lower().endswith(".wav")][:10]
    train_phns = [phn for phn in os.listdir(
        train_phn_dir) if phn.lower().endswith(".phn")][:10]


    wav_base_path = train_wav_dir
    phn_base_path = train_phn_dir
    wav_files = train_wavs
    phn_files = train_phns
    sr=16000
    time_window=0.025

    ret_x = ret_y = None
    for wav_file, phn_file in zip(wav_files, phn_files):
        print(os.path.join(wav_base_path, wav_file))

        # get amplitude graph
        x, sr = librosa.load(os.path.join(wav_base_path, wav_file), sr=sr)
        print(x.shape)

        # get power-spectrogram (a power-spectrogram is just the amplitude squared)
        # n_fft: length of the windowed signal after padding with zeros. 512 is recommended for speech processing 
        # hop_length: number of audio samples between adjacent STFT columns.
        # win_length: Each frame of audio is windowed by window() of length win_length and then padded with zeros to match n_fft.
        n_fft = 512
        hop_length = 80
        win_length = int(sr * time_window)
        D = librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        mag = np.abs(D)**2

        # compute the mel-spectrogram
        S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # convert from power-spectrogram to decibel units
        S_db = librosa.power_to_db(S)


        # create phoneme labelsphn_file = wav_file.replace("WAV.wav", "PHN").replace("wav", "PHN")
        phn2idx, _ = load_vocab()
        num_timesteps = S_db.shape[1]
        phns = np.zeros(shape=(num_timesteps,))
        for line in open(os.path.join(phn_base_path, phn_file), 'r').read().splitlines():
            start_point, _, phn = line.split()
            bnd = int(start_point) // hop_length
            print(bnd)
            phns[bnd:] = phn2idx[phn]

        print("S_db", S_db.shape)
        print("phns", phns.shape)

        duration = 2
        n_timesteps = (duration * sr) // hop_length + 1
        print(n_timesteps)
        # Padding or crop
        S_db = librosa.util.fix_length(S_db, n_timesteps, axis=1)
        phns = librosa.util.fix_length(phns, n_timesteps, axis=0)


        print("S_db", S_db.shape)
        print("phns", phns.shape)

        if ret_x is None:
            ret_x = np.expand_dims(S_db, axis=0)
        else:
            ret_x = np.append(ret_x, np.expand_dims(S_db, axis=0), axis=0)

        if ret_y is None:
            ret_y = np.expand_dims(phns, axis=0)
        else:
            ret_y = np.append(ret_y, np.expand_dims(phns, axis=0), axis=0)

    
        print(ret_x.shape, ret_y.shape)
    return ret_x, ret_y

# train_x, train_y = extract_mel_phns_target_pairs()
# print(train_x.shape, train_y.shape)
# input_shape = (
#     train_x.shape[1], train_x.shape[2])


# input_shape = (num_time_steps, num_features at each time_step)
duration = 2
sr = 16000
hop_length = 80
input_shape = ((duration * sr) // hop_length + 1, 128)

print(input_shape)


print("Build LSTM RNN model ...")
model = Sequential()

# model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.35,
#                return_sequences=True, input_shape=input_shape))
# model.add(LSTM(units=32,  dropout=0.05,
#                recurrent_dropout=0.35, return_sequences=True))
# model.add(Dense(units=len(phns), activation="softmax"))

model.add(Bidirectional(CuDNNLSTM(units=128, return_sequences=True), input_shape=input_shape))
model.add(Bidirectional(CuDNNLSTM(units=32, return_sequences=True), input_shape=input_shape))
model.add(Dense(units=len(phns), activation="softmax"))

# model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True), input_shape=input_shape))
# model.add(Attention(input_shape[0]))
# model.add(Dropout(0.2))
# model.add(Dense(400))
# model.add(ELU())
# model.add(Dropout(0.2)) 
# model.add(Dense(units=len(phns), activation='softmax'))

print("Compiling ...")
# Keras optimizer defaults:
# Adam   : lr=0.001, beta_1=0.9,  beta_2=0.999, epsilon=1e-8, decay=0.
# RMSprop: lr=0.001, rho=0.9,                   epsilon=1e-8, decay=0.
# SGD    : lr=0.01,  momentum=0.,                             decay=0.
opt = Adam()

# categorical_crossentropy: the true class is represented as a one-hot encoded vector, 
# and the closer the model’s outputs are to that vector, the lower the loss.
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])
model.summary()


print("Training ...")
batch_size = 32  # num of training examples per minibatch
num_epochs = 100
callbacks = create_callbacks(
    "phoneme_classifier_bidirectional_cudnnlstm_.h5", "logs", model)
# model.fit(
#     train_x,
#     train_y,
#     batch_size=batch_size,
#     epochs=num_epochs,
#     callbacks=callbacks
# )

# default values from Yolov3 config.json
# train_wav_dir = "./train/wav"
# train_wavs = [wav for wav in os.listdir(
#     train_wav_dir) if wav.lower().endswith(".wav")]
# train_times = 8 
# nb_epochs = 100 
# model.fit_generator(
#     generator=generate_mel_phns_target_pair(),
#     steps_per_epoch=len(train_wavs) * train_times,
#     epochs=nb_epochs,
#     verbose=1, 
#     callbacks=callbacks,
#     workers=4,
#     max_queue_size=8,
#     use_multiprocessing=True)

train_wav_dir = "./train/wav"
train_phn_dir = "./train/phn"
train_insts = create_wav_phn_file_path_pairs(train_wav_dir, train_phn_dir)

train_generator = BatchGenerator(
    instances=train_insts,
    batch_size=32,
    sr=16000,
    n_fft=512,
    hop_length=80,
    time_window=0.025,
    duration = 2,
    shuffle=True, 
)

train_times = 8 
nb_epochs = 100 
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator) * train_times,
    epochs=nb_epochs,
    verbose=1, 
    callbacks=callbacks,
    workers=4,
    max_queue_size=8)
