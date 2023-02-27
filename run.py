import numpy as np
import pickle
import librosa
import soundfile
import streamlit as st 
import os


def extract_feature(file_name, mfcc, chroma, mel):
  with soundfile.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype="float32")
    sample_rate=sound_file.samplerate
    if chroma:
      stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
      mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
      result=np.hstack((result, mfccs))
    if chroma:
      chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
      result=np.hstack((result, chroma))
    if mel:
      mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
      result=np.hstack((result, mel))
  return result

  
loaded_model = pickle.load(open('C:/STREAMLIT/modelForPrediction1.sav', 'rb')) 


Audio_file = st.file_uploader("Upload An Audio to analyze ", type=['mp3', 'wav'])
if Audio_file is not None:
    file_details = {"FileName": Audio_file.name, "FileType": Audio_file.type}

    with open(os.path.join(Audio_file.name), "wb") as f:
        f.write(Audio_file.getbuffer())
    Audio_File = Audio_file
    feature=extract_feature( Audio_File , mfcc=True, chroma=True, mel=True)
    feature=feature.reshape(1,-1)
    prediction=loaded_model.predict(feature)
    x = str(prediction[0])
    print(x)
    st.title("emotion in this audio is  "+ x)
    os.remove(Audio_file.name)
    st.success( x + "  emotion is analyzed successfully")
    


 
