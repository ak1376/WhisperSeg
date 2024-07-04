import librosa
import numpy as np
from model import WhisperSegmenterFast
from audio_utils import WhisperSegFeatureExtractor
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from tqdm import tqdm
import os

class Segmenter: 
    def __init__(self, sr, segmenter, feature_extractor, min_frequency, spec_time_step, min_segment_length, eps, num_trials, wavfiles) -> None:
        self.sr = sr
        self.segmenter = segmenter
        self.feature_extractor = feature_extractor
        self.min_frequency = min_frequency
        self.spec_time_step = spec_time_step
        self.min_segment_length = min_segment_length
        self.eps = eps
        self.num_trials = num_trials

        self.wavfiles = wavfiles

        self.onset_list = []
        self.offset_list = []

    def segment_song(self, wavfile_path):

        onset_list = []
        offset_list = []

        audio, _ = librosa.load(wavfile_path, sr = self.sr)

        prediction = self.segmenter.segment(audio, sr = sr, min_frequency = self.min_frequency, spec_time_step = self.spec_time_step,
                        min_segment_length = self.min_segment_length, eps = self.eps, num_trials = self.num_trials)
        
        onset_list.append(prediction['onset'])
        offset_list.append(prediction['offset'])

        return self.sr*np.array(onset_list), self.sr*np.array(offset_list)

    
    def silencer(self, wavfile, samples_onsets, samples_offsets):
        '''
        This function will take a noisy wavfile and find the samples according to the samples_onsets and samples_offsets arrays. I will then replace all samples outside of these regions with zeros (silences)
        '''

        # Flatten the arrays to ensure they are one-dimensional
        samples_onsets = np.array(samples_onsets).flatten()
        samples_offsets = np.array(samples_offsets).flatten()

        # Create an array of silence with the same shape and data type as the input wavfile
        silenced_audio = np.zeros_like(wavfile)

        # Copy detected regions to the silenced array
        for start, end in zip(samples_onsets, samples_offsets):
            # Explicitly convert start and end to integers to avoid indexing errors
            start_idx = int(start)
            end_idx = int(end)
            silenced_audio[start_idx:end_idx] = wavfile[start_idx:end_idx]

        return silenced_audio
    
    def create_sonogram(self, audio):
        sonogram = self.feature_extractor(audio, sampling_rate=self.sr, padding = "do_not_pad" )["input_features"][0]

        return sonogram
    

# Use of object
sr = 32000
min_frequency = 0
spec_time_step = 0.001
min_segment_length = 0.005
eps = 0.01
num_trials = 3

# audio_file = "/media/akapoor/Extreme SSD/USA5207/38/USA5207_45131.25881743_7_24_7_11_21.wav"
# audio_file = '/home/akapoor/Desktop/llb16_0852_2018_05_08_08_28_04.wav'
# audio, _ = librosa.load( audio_file, sr = sr )

segmenter = WhisperSegmenterFast( "nccratliri/whisperseg-canary-ct2", device="cpu" )

# Default values
window_size = 15
spec_width = 1000
min_frequency = 0
max_frequency = None

feature_extractor = WhisperSegFeatureExtractor(sr, window_size / spec_width, min_frequency, max_frequency )

# I want to do this operation on all of Rose's data

bird_paths = '/Users/AnanyaKapoor/Desktop/USA5207'

bird_filepaths = []

for filename in os.listdir(bird_paths):
    if filename == '.DS_Store':
        continue
    file_path = os.path.join(bird_paths, filename)
    bird_filepaths.append(file_path)

for bird_path in bird_filepaths:
    all_days = bird_path

    day_filepaths = []

    for filename in os.listdir(all_days):
        file_path = os.path.join(all_days, filename)
        day_filepaths.append(file_path)

    # Extract bird name
    # Split the path by '/'
    parts = all_days.split('/')

    # Get the last element of the list
    bird_name = parts[-1]
    
    os.makedirs(f'png_files/{bird_name}', exist_ok=True)
    os.makedirs(f'new_wav_files/{bird_name}', exist_ok=True)
    

    seg_jawn = Segmenter(sr = sr, segmenter = segmenter, feature_extractor=feature_extractor, min_frequency=min_frequency, spec_time_step=spec_time_step, min_segment_length= min_segment_length, eps = eps, num_trials=num_trials, wavfiles= day_filepaths)

    for i in tqdm(np.arange(len(seg_jawn.wavfiles))):
        audio_path = seg_jawn.wavfiles[i]
        audio, _ = librosa.load(audio_path, sr = sr )
        sample_onsets, sample_offsets = seg_jawn.segment_song(audio_path)
        silenced_audio = seg_jawn.silencer(audio, sample_onsets, sample_offsets)

        orig_spec = seg_jawn.create_sonogram(audio)
        silenced_spec = seg_jawn.create_sonogram(silenced_audio)

        # Plotting (for every 50th wav file)

        if i%50 == 0:

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Original audio spectrogram
            axes[0].imshow(orig_spec, origin='lower', cmap='viridis')
            axes[0].set_title('Original Audio Spectrogram')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Frequency Bin')

            # Silenced audio spectrogram
            axes[1].imshow(silenced_spec, origin='lower', cmap='viridis')
            axes[1].set_title('Spectrogram from Audio with Noise Removed')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Frequency Bin')

            # Display the plot
            plt.tight_layout()

            parts = day_path.split('/')
            day_value = parts[-1]

            os.makedirs(f'png_files/{bird_name}/day_{day_value}', exist_ok=True)

            # Split the path by '/'
            parts = audio_path.split('/')

            # Get the last element of the list
            last_part = parts[-1]

            song_name = last_part.replace('.wav', '.png')

            plt.savefig(f'png_files/{bird_name}/day_{day_value}/{song_name}')
            plt.close()

        
        # Split the path by '/'
        parts = audio_path.split('/')

        # Get the last element of the list
        last_part = parts[-1]

        os.makedirs(f'new_wav_files/{bird_name}/day_{day_value}', exist_ok=True)

        write(f'new_wav_files/{bird_name}/day_{day_value}/{last_part}', sr, silenced_audio)






















# samples_onsets, samples_offsets = seg_jawn.segment_song(audio_file)

# silenced_audio = seg_jawn.silencer(audio, samples_onsets, samples_offsets)

# orig_spec = seg_jawn.create_sonogram(audio)
# silenced_spec = seg_jawn.create_sonogram(silenced_audio)

# # Plotting
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# # Original audio spectrogram
# axes[0].imshow(orig_spec, origin='lower', cmap='viridis')
# axes[0].set_title('Original Audio Spectrogram')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Frequency Bin')

# # Silenced audio spectrogram
# axes[1].imshow(silenced_spec, origin='lower', cmap='viridis')
# axes[1].set_title('Silenced Audio Spectrogram')
# axes[1].set_xlabel('Time')
# axes[1].set_ylabel('Frequency Bin')

# # Display the plot
# plt.tight_layout()
# plt.show()
