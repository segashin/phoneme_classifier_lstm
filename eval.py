import numpy as np
from keras.models import load_model

from utils import create_wav_phn_file_path_pairs
from generator import BatchGenerator

model = load_model("./phoneme_classifier_bidirectional_cudnnlstm.h5")

test_wav_dir = "./test/wav"
test_phn_dir = "./test/phn"
test_insts = create_wav_phn_file_path_pairs(test_wav_dir, test_phn_dir)

test_generator = BatchGenerator(
    instances=test_insts,
    batch_size=32,
    sr=16000,
    n_fft=512,
    hop_length=80,
    time_window=0.025,
    duration = 2,
    shuffle=False, 
)

predictions = model.predict_generator(
    generator=test_generator,
    verbose=1, 
    workers=4,
    max_queue_size=8)

print(predictions.shape)
# (640, 401, 61)

correct_timesteps = 0
total_timesteps = 0
for i in range(test_generator.size()):
    gt_phns = test_generator.load_phns(i)

    gt_phn_idxs = gt_phns.argmax(2)
    pred_phn_idxs = predictions[i, :, :].argmax(1)

    print(np.count_nonzero(pred_phn_idxs == gt_phn_idxs))
    correct_timesteps += np.count_nonzero(pred_phn_idxs == gt_phn_idxs)
    total_timesteps += pred_phn_idxs.shape[0]

print(f"total accuracy is {correct_timesteps / total_timesteps}")



