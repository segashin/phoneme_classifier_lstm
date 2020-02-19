import cv2
import copy
import numpy as np
import librosa
import librosa.display
from keras.utils import Sequence

# For plotting headlessly
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas




# instances: list of wav_file_path and phn_file_path tuples [("<wav_file_path>", "<phn_file_path>"), ...]
class BatchGenerator(Sequence):
    def __init__(self, 
        instances,      
        batch_size=1,
        sr=16000,
        n_fft=512,
        hop_length=80,
        time_window=0.025,
        duration = 2,
        shuffle=True, 
    ):
        phns = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl',
                'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi',
                'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh',
                'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
                'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']

        self.instances          = instances
        self.batch_size         = batch_size
        self.sr                 = sr
        self.n_fft              = n_fft
        self.hop_length         = hop_length
        self.win_length         = int(sr * time_window)
        self.duration           = duration
        self.n_timesteps        = (duration * sr) // hop_length + 1
        self.shuffle            = shuffle
        
        if shuffle: np.random.shuffle(self.instances)

        self.num_phns = len(phns)
        self.phn2idx = {phn: idx for idx, phn in enumerate(phns)}
        self.idx2phn = {idx: phn for idx, phn in enumerate(phns)}
            
    def __len__(self):
        return int(np.ceil(float(len(self.instances))/self.batch_size))           

    def __getitem__(self, idx):
        # determine the first and the last indices of the batch
        l_bound = idx*self.batch_size
        r_bound = (idx+1)*self.batch_size

        if r_bound > len(self.instances):
            r_bound = len(self.instances)
            l_bound = r_bound - self.batch_size

        train_X_batch = None
        train_Y_batch = None

        # print(l_bound, r_bound)
        # do the logic to fill in the inputs and the output
        for train_instance in self.instances[l_bound:r_bound]:
            # S_db is mel spectrogram in decibels: shape is (1, num_time_steps, num_features_per_time_step) 
            # phnms is phonemes represented as one hot encoded vectors for each time_step: shape is (1, num_time_steps, num_phonemes)
            S_db, phns = self._get_mel_phns_target_pair(train_instance[0], train_instance[1])
            
            # stack the results
            if train_X_batch is None:
                train_X_batch = S_db
            else:
                train_X_batch = np.append(train_X_batch, S_db, axis=0)

            if train_Y_batch is None:
                train_Y_batch = phns
            else:
                train_Y_batch = np.append(train_Y_batch, phns, axis=0)

        # print(train_X_batch.shape, train_Y_batch.shape)
        return train_X_batch, train_Y_batch

    # returns mel spectrogram and one hot encoded label for correct phoneme at each timestep
    def _get_mel_phns_target_pair(self, wav_file_path, phn_file_path):
        # print(wav_file_path, phn_file_path)
        # get amplitude graph
        x, _ = librosa.load(wav_file_path, self.sr)
        # print(x.shape)

        # get power-spectrogram (a power-spectrogram is just the amplitude squared)
        # n_fft: length of the windowed signal after padding with zeros. 512 is recommended for speech processing 
        # hop_length: number of audio samples between adjacent STFT columns.
        # win_length: Each frame of audio is windowed by window() of length win_length and then padded with zeros to match n_fft.
        D = librosa.stft(y=x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        mag = np.abs(D)**2

        # compute the mel-spectrogram
        S = librosa.feature.melspectrogram(S=D, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        # convert from power-spectrogram to decibel units
        S_db = librosa.power_to_db(S)

        temp_num_timesteps = S_db.shape[1]
        phns = np.zeros(shape=(temp_num_timesteps, self.num_phns))
        for line in open(phn_file_path, 'r').read().splitlines():
            start_point, _, phn = line.split()
            bnd = int(start_point) // self.hop_length

            # one hot encoding of correct phoneme
            one_hot = np.zeros((self.num_phns))
            one_hot[self.phn2idx[phn]] = 1.0
            phns[bnd:] = one_hot

        
        # print(n_timesteps)
        # Padding or crop
        S_db = librosa.util.fix_length(S_db, self.n_timesteps, axis=1)
        phns = librosa.util.fix_length(phns, self.n_timesteps, axis=0)

        # if wav_file_path == "./train/wav/SI2012.WAV.wav":
        #     print("S_db", S_db.shape)
        #     print("phns", phns.shape)
        #     plt.figure(figsize=(15, 5))
        #     # librosa.display.specshow(
        #     #     S_db, sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        #     librosa.display.specshow(
        #         phns.transpose(), sr=self.sr, hop_length=self.hop_length, x_axis='time', y_axis='mel')
        #     plt.colorbar(format='%+2.0f dB')
        #     plt.savefig('phns1.png')

        S_db = S_db.transpose()

        # reshape to stack later 
        S_db = np.reshape(S_db, (1, S_db.shape[0], S_db.shape[1]))
        phns = np.reshape(phns, (1, phns.shape[0], phns.shape[1]))

        # print("S_db", S_db.shape)
        # print("phns", phns.shape)

        return S_db, phns

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.instances)
            
    def num_phns(self):
        return num_phns

    def size(self):
        return len(self.instances)    
    
    def load_phns(self, i):
        print(self.instances[i][0])
        _, phns = self._get_mel_phns_target_pair(self.instances[i][0], self.instances[i][1])
        return phns
