import numpy as np
import pickle
import librosa
import glob

def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    # this function first normalizes audio data
    #calculates the amplitude of each frame
    #silence_threshold is used to flip the silence part
    #the number of silence frame is returned.
    #trim_ms is the counter
    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms

def feature_extract(path):

    # extract feature from the wav
    window_size = 2048
    hop_size = window_size/2
    data = []

    #read file
    files = glob.glob(path)
    np.random.shuffle(files)
    for filename in files:
        print(filename)
        music, sr= librosa.load(filename)
        #remove the silence time from the beginning and ending of the file
        start_trim = detect_leading_silence(music)
        end_trim = detect_leading_silence(np.flipud(music))

        duration = len(music)
        trimmed_sound = music[start_trim:duration-end_trim]
        # the sound without silence

        #use the Mel-frequency cepstral coefficients as features for classification
        mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
        aver = np.mean(mfccs, axis = 1)
        # the feature variable is the mean vector of mfccs over time and reshaped to 20
        feature = aver.reshape(20)

        #store label and feature
        #the output should be a list
        #label and feature, corresponds one by one
        #feature.append(aver)
        instrument_name = filename[65:].split('_')[0]
        data.append([filename, feature, instrument_name])
        #data = np.vstack((data, data2))
        # print data
    return data


def main():
    #extract features from training data set and dump them to disk for later usage
    data = feature_extract('C:\\Users\\ChenWenshi\\Documents\\GitHub\\umich_ds_cc_2017\\train_data\\*.wav')
    with open("data.dat", "wb") as f:
        pickle.dump(data, f)
if __name__ == '__main__':
     main()
