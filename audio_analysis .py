import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sb
import soundfile
import os
import librosa 
import librosa.display 
import IPython.display as ipd
from glob import glob
import csv
from csv import writer
from pydub import AudioSegment
from pydub.utils import make_chunks
import os

def audio_analyse(filename):

    # First load .wav audio and cut into same segments as video analysis (default 1s)

    myaudio = AudioSegment.from_file(f"working_files/{filename}.wav" , "wav") 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

    # Export all of the individual chunks as wav files

    for i, chunk in enumerate(chunks):
        # chunk_name = "depressed_chunk{0}.wav".format(i)
        chunk_name = filename + "{0}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export(chunk_name, format="wav")

    #DataFlair - Extract features (mfcc, chroma, mel) from a sound file
    #mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    #chroma: Pertains to the 12 different pitch classes
    #librosa.feature.chroma_stft returns a chroma spectrogram (or chromogram). It has shape (12, n_frames). 12 for each of the 12 semitones in an octave C,C#,D..., B. Each bin in the chroma spectrogram represents the average energy of that semitone (across all octaves).
    def extract_feature(file_name, chroma):
        data_y,sample_rate = librosa.load(file_name)
        if chroma:
            stft=np.abs(librosa.stft(data_y))
        result=np.array([])
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))   
        return result

    def load_data(audio_train):
        x=[]
        for file in audio_train:
            feature=extract_feature(file, chroma=True)
            x.append(feature)
        return np.array(x)

    column_names = ['Pitch 1', 'Pitch 2', 'Pitch 3', 'Pitch 4', 'Pitch 5', 'Pitch 6', 'Pitch 7', 'Pitch 8','Pitch 9', 'Pitch 10', 'Pitch 11', 'Pitch 12']
    audio_files= glob('../working_files/audio_chunks/*.wav')
    audio_data= load_data(audio_files)
    audio_df = pd.DataFrame(audio_data)
    return(audio_df)