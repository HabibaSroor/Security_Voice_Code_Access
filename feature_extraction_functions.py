import numpy as np
import librosa
import librosa.display


def mfcc_feature_extractor(audio,sampleRate):
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCC) features from audio.

    Parameters:
    - audio (numpy.ndarray): Audio signal.
    - sampleRate (int): Sampling rate of the audio.

    Returns:
    - numpy.ndarray: Extracted MFCC features.
    """
    mfccsFeatures = librosa.feature.mfcc(y=audio, sr=sampleRate, n_mfcc=40)
    mfccsScaledFeatures = np.mean(mfccsFeatures.T,axis=0)
    print(f"mfcc {mfccsFeatures.shape}")
    print(f"mfcc scaled  {mfccsScaledFeatures.shape}")
    return mfccsScaledFeatures

def contrast_feature_extractor(audio,sampleRate):
    """
    Extracts spectral contrast features from audio.

    Parameters:
    - audio (numpy.ndarray): Audio signal.
    - sampleRate (int): Sampling rate of the audio.

    Returns:
    - numpy.ndarray: Extracted spectral contrast features.
    """
    stft = np.abs(librosa.stft(audio))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sampleRate) 
    contrastScaled = np.mean(contrast.T,axis=0)
    print(f"contrasr {contrast.shape}")
    print(f"contsrast scaled  {contrastScaled.shape}")
    
    return contrastScaled

def tonnetz_feature_extractor(audio,sampleRate):
    """
    Extracts tonnetz features from audio.

    Parameters:
    - audio (numpy.ndarray): Audio signal.
    - sampleRate (int): Sampling rate of the audio.

    Returns:
    - numpy.ndarray: Extracted tonnetz features.
    """
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sampleRate)
    tonnetzScaled = np.mean(tonnetz.T,axis=0)
    print(f"tonnetz {tonnetz.shape}")
    print(f"tonnetz scaled  {tonnetzScaled.shape}")
    
    return tonnetzScaled

def centroid_feature_extractor(audio,sampleRate):
    """
    Extracts spectral centroid features from audio.

    Parameters:
    - audio (numpy.ndarray): Audio signal.
    - sampleRate (int): Sampling rate of the audio.

    Returns:
    - numpy.ndarray: Extracted spectral centroid features.
    """
    centroid = librosa.feature.spectral_centroid(y=audio,sr=sampleRate) 
    centroidScaled = np.mean(centroid.T,axis=0)
    print(f"centroid {centroid.shape}")
    print(f"centroid scaled  {centroidScaled.shape}")
    
    return centroidScaled

def chroma_feature_extractor(audio,sampleRate):
    """
    Extracts chroma features from audio.

    Parameters:
    - audio (numpy.ndarray): Audio signal.
    - sampleRate (int): Sampling rate of the audio.

    Returns:
    - numpy.ndarray: Extracted chroma features.
    """
    stft = np.abs(librosa.stft(audio))
    chroma = librosa.feature.chroma_stft(S=stft,sr=sampleRate)
    chromaScaled = np.mean(chroma.T,axis=0)
    print(f"centroid {chroma.shape}")
    print(f"centroid scaled  {chromaScaled.shape}")
    
    return chromaScaled

def features_extractor(file):
    """
    Extracts various features from an audio file.

    Parameters:
    - file (str): File path of the audio.

    Returns:
    - list: List containing concatenated features.
    """
    features=[]
    audio, sampleRate = librosa.load(file) 
    mfcc=mfcc_feature_extractor(audio,sampleRate)
    contrast = contrast_feature_extractor(audio,sampleRate)
    tonnetz = tonnetz_feature_extractor(audio,sampleRate)
    chroma = chroma_feature_extractor(audio,sampleRate)

    features.append([mfcc,contrast,tonnetz,chroma])
    features[0] = np.concatenate((features[0][0],features[0][1],features[0][2],features[0][3]))
    return features

