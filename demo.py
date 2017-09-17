import numpy as np
import sklearn
import librosa
import glob
from sklearn import svm
from sklearn.externals import joblib
from sklearn import decomposition
from sklearn import neighbors
import preprocessing
import classify
import csv

def feature_extract(path):
    audio_filenames = glob.glob(path)
    data = []
    for audio_filename in audio_filenames:
        print(audio_filename)
        music, sr= librosa.load(audio_filename)
        start_trim = preprocessing.detect_leading_silence(music)
        end_trim = preprocessing.detect_leading_silence(np.flipud(music))

        duration = len(music)
        trimmed_sound = music[start_trim:duration-end_trim]

        mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
        aver = np.mean(mfccs, axis = 1)
        audio_feature = aver.reshape(20)
        data.append(audio_feature)
    return np.array(data)


def main():
    path = 'C:\\Users\\ChenWenshi\\Downloads\\umich_ds_cc_2017-master\\test_data\\*.wav'
    demo_data = feature_extract(path) # extract features from all wav files in the test dataset
    print('processed data.')
    result = []
    model_params = {
        'pca_n': 10,
        'knn_k': 5,
        'knn_metric': 'minkowski'
    }
    #  train_and_test(data, [model_params, 'svc'])
    model = classify.load_model(model_params) # load trained data
    pre = classify.predict(model, demo_data, [model_params, 'svc']) # predict with SVC method
    audio_filenames = glob.glob(path)
    for i in range(len(audio_filenames)):
        file = audio_filenames[i][64:]
        result.append((file,pre[i]))
    f = open('result.csv', 'w')
    for row in result:
        f.write(row[0]+','+row[1]+'\n')
    f.close()

if __name__ == '__main__':
    main()
