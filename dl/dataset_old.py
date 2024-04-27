import os
from torch.utils.data import Dataset
from os.path import join
import json
import librosa
import numpy as np
import torch
from config import hop_length, sample_rate

class SongAndBeatmapsDataset(Dataset):    
    def __init__(self, root='dataset',level='ExpertPlus', train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.level = level
        
        self.songs_folder = join(self.root, 'songs')
        self.color_notes_folder = join(self.root, 'beatmaps', 'color_notes')
        self.bomb_notes_folder = join(self.root, 'beatmaps', 'bomb_notes')
        self.obstacles_folder = join(self.root, 'beatmaps', 'obstacles')
        
        self.labels = self.get_labels()
        
        with open(join(self.root, 'song_levels.json'), 'r') as f:
            self.song_levels = json.load(f)

    def load_mel_spectrogram(self, song_filename):
        mel_spectrogram = np.load(join(self.songs_folder, song_filename))
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
    
    def load_beats(self, beats_filename):
        beats = np.load(join(self.color_notes_folder,self.level, beats_filename))
        return beats
    
    def get_labels(self):
        labels = np.load(join(self.color_notes_folder, 'labels.npy')).tolist()
        return labels
    
    def get_timestamps(self, mel_spectrogram, beats, bpm):
        timestamps = librosa.times_like(mel_spectrogram, sr=sample_rate, hop_length=hop_length)
        
        timestamp_features = []
        timestamp_labels = []
        for obj in beats:
            beat_time = obj[1]

            beat_time_to_sec = beat_time / bpm * 60

            closest_frame_idx = np.argmin(np.abs(timestamps - beat_time_to_sec))
            mel_feature_for_object = mel_spectrogram[:, closest_frame_idx]
            
            note_label = [int(obj[2]), int(obj[3]), int(obj[4]), int(obj[5])]
            find_label = self.labels.index(note_label)

            
            timestamp_features.append(mel_feature_for_object)
            timestamp_labels.append(find_label)
        
        timestamp_features = np.array(timestamp_features)
        timestamp_labels = np.array(timestamp_labels)
        
        timestamp_features = torch.from_numpy(timestamp_features).float()
        timestamp_labels = torch.from_numpy(timestamp_labels).long()
        
        return timestamp_features, timestamp_labels
    
    def combine_beats_and_mel(self, timestamp_features, timestamp_labels):

        list(zip(timestamp_features, timestamp_labels))

    def __getitem__(self, index):
        filename = os.listdir(join(self.color_notes_folder, self.level))[index]
        
        song = filename.split('_')[0]
        
        bpm = self.song_levels[song]['bpm']
        
        mel_spectrogram = self.load_mel_spectrogram(filename)
        beats = self.load_beats(filename)
        
        timestamp_features, timestamp_labels = self.get_timestamps(mel_spectrogram, beats, bpm)
        
        return timestamp_features, timestamp_labels

    def __len__(self):
        return len(os.listdir(join(self.color_notes_folder, self.level)))

if __name__ == '__main__':
    from tqdm.auto import tqdm