import os
import time
import numpy
import obspy

import warnings
warnings.filterwarnings("ignore")

metadata = './events.csv'
earthquake_data = './earthquakes_db'

dataset = list()

checkpoint = time.time()

with open(metadata) as metadata_file:
    for line in metadata_file:
        content = line.strip().split(',')
        if content[0] != 'event_id':
            try:
                data = dict()
                #
                data['id'] = content[0]
                data['date'] = content[1]
                data['latitude'] = float(content[2])
                data['longitude'] = float(content[3])
                data['depth'] = float(content[4])
                data['magnitude'] = float(content[5])
                #
                path_signal_E = os.path.join(earthquake_data, '%s_NNA_E.mseed' % data['id'])
                path_signal_N = os.path.join(earthquake_data, '%s_NNA_N.mseed' % data['id'])
                path_signal_Z = os.path.join(earthquake_data, '%s_NNA_Z.mseed' % data['id'])
                #
                data['signal_E'] = obspy.read(path_signal_E)[0].data
                data['signal_N'] = obspy.read(path_signal_N)[0].data
                data['signal_Z'] = obspy.read(path_signal_Z)[0].data

                if len(data['signal_E']) and len(data['signal_N']) and len(data['signal_Z']) == 12000:
                    dataset.append(data)
            except:
                pass

print('Dataset read (Time: %.3fs)' % (time.time() - checkpoint))
print('Number of samples: %d' % (len(dataset)))

def plot_signal(index=0):
    import pandas
    import seaborn
    import matplotlib.pyplot as plt

    x = numpy.array(dataset[index]['signal_E'])
    y = numpy.array(dataset[index]['magnitude'])
    d = numpy.array(dataset[index]['depth'])
    df = pandas.DataFrame(dict(time=numpy.arange(12000), value=x, magnitude=y, depth=d))
    seaborn.relplot(x='time', y='value', kind='line', col='magnitude', row='depth', data=df)
    plt.tight_layout()
    plt.show()

def plot_magnitude_distribution():
    import seaborn
    import matplotlib.pyplot as plt

    x = numpy.array([data['magnitude'] for data in dataset])
    seaborn.kdeplot(x, shade=True)
    plt.tight_layout()
    plt.show()

def perform_holdout():
    from keras.models import Sequential
    from keras.layers import Dense, LSTM
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline

    X = list()
    Y = list()

    for data in dataset[:1000]:
        X.append(data['signal_E'])
        Y.append(data['magnitude'])

    X = numpy.array(X)
    Y = numpy.array(Y)

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X = scaler.fit_transform(X)

    # X = numpy.reshape(X, (X.shape[0], X.shape[1], 1))

    def baseline_mlp():
        model = Sequential()
        model.add(Dense(1000, input_dim=12000, kernel_initializer='normal', activation='relu'))
        model.add(Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def baseline_lstm():
        model = Sequential()
        model.add(LSTM(4, input_shape=(12000, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    seed = 7
    numpy.random.seed(seed)
    estimator = KerasRegressor(build_fn=baseline_mlp, epochs=100, batch_size=10, verbose=1)

    kfold = KFold(n_splits=10, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold, verbose=1)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

perform_holdout()