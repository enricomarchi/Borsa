import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.subplots as sp

from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l1, l2
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

def addestramento_classificazione(model, features, target, learning_rate, batch_size, look_back):    
    X, Y = converti_in_XY(features, target, look_back)
    Y = np.where(Y, 1, 0)
    class_weights = compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(class_weights)     

    if model == 0:
        model = keras.Sequential([
            layers.LSTM(256, input_shape=(X.shape[1], X.shape[2]), kernel_regularizer=l2(0.01), return_sequences=True),
            layers.Dropout(0.2),
            #layers.LSTM(500, kernel_regularizer=l2(0.01), return_sequences=True),  # Nuovo layer LSTM 1
            #layers.Dropout(0.2),
            #layers.LSTM(128, kernel_regularizer=l2(0.01), return_sequences=True),  # Nuovo layer LSTM 2
            #layers.Dropout(0.2),
            layers.LSTM(128, kernel_regularizer=l2(0.01)),  # Ultimo layer LSTM
            layers.Dropout(0.2),
            layers.Dense(50, kernel_regularizer=l2(0.01)),  
            layers.Dropout(0.2),
            layers.Dense(1, kernel_regularizer=l2(0.01), activation='sigmoid')
        ])
    custom_optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=custom_optimizer,
        loss='binary_crossentropy',
        metrics=[
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            'accuracy'
        ]
    )

    #rus = RandomUnderSampler(sampling_strategy='auto')
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X.shape
    X = X.reshape((n_samples, n_timesteps * n_features)) # trasforma in 2D
    #X, Y = rus.fit_resample(X, Y) # undersampling
    X = scaler.fit_transform(X) #standard scaler
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # split in set di train e di test
    X_train = X_train.reshape((-1, n_timesteps, n_features)) # ripristina in 3D
    X_test = X_test.reshape((-1, n_timesteps, n_features)) # ripristina in 3D
    
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)
    if batch_size == "max":
        batch_size = len(X_train)
    #n = len(X_train)
    #weights = np.linspace(1, n, n)

    model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        validation_data=(X_test, Y_test),
        verbose=1,
        shuffle=False,
        #sample_weight=weights,
        callbacks=[
            early_stopping
        ]
    )

    loss, precision, recall, auc, accuracy = model.evaluate(X_test, Y_test)
    print(f"\033[42mLoss: {loss}, Precision: {precision}, Recall: {recall}, AUC: {auc}, Accuracy: {accuracy}\033[0m")
    with open("result.txt", "a") as f:
        f.write(f"Loss: {loss}, Precision: {precision}, Recall: {recall}, AUC: {auc}, Accuracy: {accuracy}\n")
    return model


def features_e_target_classificazione(df):
    features = df[[
        "Open", 
        "High", 
        "Low", 
        "Close", 
        #"Adj Close", 
        #"Volume",
        #"EMA_5",
        #"EMA_20",
        #"EMA_50",
        #"EMA_100",
        #"PSAR",
        #"PSARaf",
        #"PSARr",
        #"MACD",
        #"MACDh",
        #"MACDs",
        #"TSI",
        #"TSIs",
        #"SUPERT",
        #"SUPERTd",
        #"ADX",
        #"DM_OSC",
        #"TRIX",
        #"AROONOSC",
        #"ATR",
        #"VTX_OSC",
        #"VI_OSC",
        #"HLC3",
    ]]        
    target = (df["perc_Close_20d"] >= 30) & (df["Perc_Drawdown_20d"] < 10)
    return features, target


def addestramento_regressione(model, features, target, learning_rate, batch_size, look_back):    
    X, Y = converti_in_XY(features, target, look_back)

    if model == 0:
        model = keras.Sequential([
            layers.LSTM(50, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
            #layers.Dropout(0.2),
            layers.LSTM(50,  return_sequences=False),
            #layers.Dropout(0.2),
            #layers.Dense(25),
            #layers.Dropout(0.2),
            layers.Dense(1)  # Linear activation (default)
        ])

    custom_optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=custom_optimizer,
        loss='mean_squared_error',
    )

    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    n_samples, n_timesteps, n_features = X.shape
    X = X.reshape((n_samples, n_timesteps * n_features))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    #X_scaler.fit(X_train)
    #Y_scaler.fit(Y_train.reshape(-1, 1))
    #X_train = X_scaler.transform(X_train)
    #X_test = X_scaler.transform(X_test)
    #Y_train = Y_scaler.transform(Y_train.reshape(-1, 1))    
    #Y_test = Y_scaler.transform(Y_test.reshape(-1, 1))
    
    X_train = X_train.reshape((-1, n_timesteps, n_features))
    X_test = X_test.reshape((-1, n_timesteps, n_features))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # Changed from 'val_accuracy' to 'val_loss'

    if batch_size == "max":
        batch_size = len(X_train)

    model.fit(
        X_train, Y_train,
        epochs=5,
        batch_size=batch_size,
        validation_data=(X_test, Y_test),
        verbose=1,
        shuffle=False,
        callbacks=[
            early_stopping
        ]
    )

    loss = model.evaluate(X_test, Y_test)  # Changed metrics
    print(f"\033[42mLoss: {loss}\033[0m")  # Changed metrics
    
    return model, X_scaler, Y_scaler


def features_e_target_regressione(df):
    features = df[[
        "Open", 
        "High", 
        "Low", 
        "Close", 
        #"Adj Close", 
        #"Volume",
        #"EMA_5",
        "EMA_20",
        "EMA_50",
        #"EMA_100",
        #"PSAR",
        #"PSARaf",
        #"PSARr",
        #"MACD",
        #"MACDh",
        #"MACDs",
        #"TSI",
        #"TSIs",
        #"SUPERT",
        #"SUPERTd",
        #"ADX",
        #"DM_OSC",
        #"TRIX",
        #"AROONOSC",
        #"ATR",
        #"VTX_OSC",
        #"VI_OSC",
        #"HLC3",
    ]]        
    target = df["Close_1d"]
    return features, target


def converti_in_XY(features, target, look_back):
    X, Y = [], []
    for i in range(look_back, len(features)):
        X.append(features.iloc[i - look_back:i].values)
        Y.append(target[i])
    X, Y = np.array(X), np.array(Y)
    return X, Y    

def crea_indicatori(df):
    psar = ta.psar(high=df["High"], low=df["Low"], close=df["Close"], af0=0.02, af=0.02, max_af=0.2)
    psar["PSAR"] = psar["PSARl_0.02_0.2"].combine_first(psar["PSARs_0.02_0.2"])
    psar.drop(["PSARl_0.02_0.2", "PSARs_0.02_0.2"], axis=1, inplace=True)
    macd = ta.macd(close=df["Close"], fast=20, slow=50, signal=9)
    tsi = ta.tsi(close=df["Close"], fast=13, slow=25)
    supertrend = ta.supertrend(high=df["High"], low=df["Low"], close=df["Close"], length=20, multiplier=3)
    supertrend.drop(["SUPERTl_20_3.0", "SUPERTs_20_3.0"], axis=1, inplace=True)
    ema5 = ta.ema(close=df["Close"], length=5)
    ema20 = ta.ema(close=df["Close"], length=20)
    ema50 = ta.ema(close=df["Close"], length=50)
    ema100 = ta.ema(close=df["Close"], length=100)
    adx = ta.adx(high=df["High"], low=df["Low"], close=df["Close"], length=20)
#    roc = ta.roc(close=df["Close"], length=10)
#    cmf = ta.cmf(high=df["High"], low=df["Low"], close=df["Close"], volume=df['Volume'], length=10)
    trix = ta.trix(close=df['Close'], length=18)
#    klinger = ta.kvo(high=df["High"], low=df["Low"], close=df["Close"], volume=df['Volume'], short=34, long=55)
    vi = ta.vortex(high=df['High'], low=df['Low'], close=df['Close'], length=20)
    aroon = ta.aroon(high=df['High'], low=df['Low'], close=df['Close'], length=20)
    nvi = ta.nvi(close=df['Close'], volume=df['Volume'])
    pvi = ta.pvi(close=df['Close'], volume=df['Volume'])
#    vhf = ta.vhf(close=df['Close'], length=20)
    atr = ta.atr(high=df['High'], low=df['Low'], close=df['Close'])

    df = pd.concat([df, ema5, ema20, ema50, ema100, psar, macd, tsi, supertrend, adx, trix, vi, aroon, nvi, pvi, atr], axis=1)

    df = __rinomina_colonne(df)

    df = __calcolo_drawdown_gain(df, 20)
    df = __calcolo_drawdown_gain(df, 50)
    df = __calcolo_drawdown_gain(df, 100)
    df["max_gain"] = df[["Perc_Max_High_Futuro_20d", "Perc_Max_High_Futuro_50d", "Perc_Max_High_Futuro_100d"]].max(axis=1)
    df["max_drawdown"] = df[["Perc_Drawdown_20d", "Perc_Drawdown_50d", "Perc_Drawdown_100d"]].min(axis=1)

    df['EMA_5_20d'] = df['EMA_5'].shift(-20)
    df['Close_20d'] = df['Close'].shift(-20)
    df['Close_1d'] = df['Close'].shift(-1)
    df['perc_EMA_5_20d'] = ((df['EMA_5_20d'] - df['EMA_5']) / df['EMA_5']) * 100
    df['perc_Close_20d'] = ((df['Close_20d'] - df['Close']) / df['Close']) * 100
    df['incrocio_verde_gialla'] = (ta.cross(df['EMA_20'], df['EMA_50'], above=True)).astype("int8")
    df["incrocio_passato_verde_gialla_10d"] = df["incrocio_verde_gialla"].rolling(10).sum()
    
    df['HLC3'] = ((df['High'] + df['Low'] + df['Close']) / 3)
    df["DM_OSC"] = (df["DMP"] - df["DMN"])
    df["VTX_OSC"] = (df["VTXP"] - df["VTXM"])
    df["VI_OSC"] = (df["PVI"] - df["NVI"])
    
    df.drop(columns=["DMP", "DMN", "VTXP", "VTXM", "PVI", "NVI", "AROOND", "AROONU"], inplace=True, axis=1)
    
    df["MaxMinRel"] = 0
    df = __trova_massimi_minimi(df, 20)   
    df = __trova_massimi_minimi(df, 50)   
    df = __trova_massimi_minimi(df, 100)   
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True, axis=0)

    return df


def __trova_massimi_minimi(df, periodo):
    mezzo_periodo = periodo // 2

    massimi_passati = df['Close'].shift(1).rolling(mezzo_periodo).max()
    massimi_futuri = df['Close'][::-1].shift(1).rolling(mezzo_periodo).max()[::-1]
    idx_massimi = (df["Close"] >= massimi_passati) & (df["Close"] >= massimi_futuri)
    df.loc[idx_massimi, "MaxMinRel"] = periodo

    minimi_passati = df['Close'].shift(1).rolling(mezzo_periodo).min()
    minimi_futuri = df['Close'][::-1].shift(1).rolling(mezzo_periodo).min()[::-1]
    idx_minimi = (df["Close"] <= minimi_passati) & (df["Close"] <= minimi_futuri)
    df.loc[idx_minimi, "MaxMinRel"] = -periodo
        
    return df

    
def __riduci_dimensioni_colonne(df):
    # Converti float64 in float32
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')

    # Converti int64 in int32
    for col in df.select_dtypes(include=['int64']).columns:
        if col != 'Volume':
            df[col] = df[col].astype('int32')

    # Converti colonne specifiche in int8
    cols_to_convert_to_int8 = [
        'PSARr',
        'SUPERTd',
        'MaxMinRel',
    ]  
    for col in cols_to_convert_to_int8:
        df[col] = df[col].astype('int8')

    # Converti colonne booleane in int8
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype('int8')
    
    return df


def __rinomina_colonne(df):
    df = df.rename(columns={
        'PSARaf_0.02_0.2': 'PSARaf',
        'PSARr_0.02_0.2': 'PSARr',
        'MACD_20_50_9': 'MACD',
        'MACDh_20_50_9': 'MACDh',
        'MACDs_20_50_9': 'MACDs',
        'TSI_13_25_13': 'TSI',
        'TSIs_13_25_13': 'TSIs',
        'SUPERT_20_3.0': 'SUPERT',
        'SUPERTd_20_3.0': 'SUPERTd',
        'ADX_20': 'ADX',
        'DMP_20': 'DMP',
        'DMN_20': 'DMN',
        'CMF_10': 'CMF',
        'TRIX_18_9': 'TRIX',
        'TRIXs_18_9': 'TRIXs',
#        'KVO_34_55_13': 'KVO',
#        'KVOs_34_55_13': 'KVOs',
#        'DCL_20_20': 'DCL',
#        'DCM_20_20': 'DCM',
#        'DCU_20_20': 'DCU',
        'VTXP_20': 'VTXP',
        'VTXM_20': 'VTXM',
        'AROOND_20': 'AROOND',
        'AROONU_20': 'AROONU',
        'AROONOSC_20': 'AROONOSC',
        'NVI_1': 'NVI',
        'PVI_1': 'PVI',
#        'VHF_20': 'VHF',
        'ATRr_14': 'ATR'
    })
    return df

    
def __calcolo_drawdown_gain(df, periodo):
    df[f"Max_High_Futuro_{periodo}d"] = df["High"].shift(-periodo).rolling(periodo).max()
    df[f"Min_Low_Futuro_{periodo}d"] = df["Low"].shift(-periodo).rolling(periodo).min()
    df[f"Drawdown_{periodo}d"] = df["Open"] - df[f"Min_Low_Futuro_{periodo}d"]
    df[f"Drawdown_{periodo}d"] = df[f"Drawdown_{periodo}d"].where(df[f"Drawdown_{periodo}d"] > 0, 0)
    df[f"Perc_Max_High_Futuro_{periodo}d"] = ((df[f"Max_High_Futuro_{periodo}d"] - df["Open"]) / df["Open"]) * 100
    df[f"Perc_Drawdown_{periodo}d"] = ((df[f"Drawdown_{periodo}d"]) / df["Open"]) * 100 
    df[f"Perc_Drawdown_{periodo}d"] = df[f"Perc_Drawdown_{periodo}d"].where(df[f"Perc_Drawdown_{periodo}d"] > 0, 0)
    df.drop(columns=[f"Max_High_Futuro_{periodo}d", f"Min_Low_Futuro_{periodo}d", f"Drawdown_{periodo}d"], axis=1, inplace=True)
    return df


def grafico_base(df): 
    close = go.Scatter(
        x = df.index,
        y = df['Close'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 0, .9)'),
        name = 'Close'
    )

    close2 = go.Scatter( # serve solo per il fill del supertrend
        x = df.index,
        y = df['Close'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        name = 'Close2'
    )

    min5 = go.Scatter(
        x = df[df['MaxMinRel'] == -5].index,
        y = df[df['MaxMinRel'] == -5]['Close'],
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(255, 0, 0, .9)'
        ),
        name = 'MinRel5'
    )
  
    max5 = go.Scatter(
        x = df[df['MaxMinRel'] == 5].index,
        y = df[df['MaxMinRel'] == 5]['Close'],
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(50, 205, 50, .9)'
        ),
        name = 'MaxRel5'
    )

    min10 = go.Scatter(
        x = df[df['MaxMinRel'] == -10].index,
        y = df[df['MaxMinRel'] == -10]['Close'],
        mode = 'markers',
        marker = dict(
            size = 15,
            color = 'rgba(255, 0, 0, .4)'
        ),
        name = 'MinRel10'
    )
  
    max10 = go.Scatter(
        x = df[df['MaxMinRel'] == 10].index,
        y = df[df['MaxMinRel'] == 10]['Close'],
        mode = 'markers',
        marker = dict(
            size = 15,
            color = 'rgba(50, 205, 50, .4)'
        ),
        name = 'MaxRel10'
    )

    min20 = go.Scatter(
        x = df[df['MaxMinRel'] == -20].index,
        y = df[df['MaxMinRel'] == -20]['Close'],
        mode = 'markers',
        marker = dict(
            size = 25,
            color = 'rgba(255, 0, 0, .2)'
        ),
        name = 'MinRel20'
    )
  
    max20 = go.Scatter(
        x = df[df['MaxMinRel'] == 20].index,
        y = df[df['MaxMinRel'] == 20]['Close'],
        mode = 'markers',
        marker = dict(
            size = 25,
            color = 'rgba(50, 205, 50, .2)'
        ),
        name = 'MaxRel20'
    )

    min60 = go.Scatter(
        x = df[df['MaxMinRel'] == -60].index,
        y = df[df['MaxMinRel'] == -60]['Close'],
        mode = 'markers',
        marker = dict(
            size = 35,
            color = 'rgba(255, 0, 0, .1)'
        ),
        name = 'MinRel60'
    )
  
    max60 = go.Scatter(
        x = df[df['MaxMinRel'] == 60].index,
        y = df[df['MaxMinRel'] == 60]['Close'],
        mode = 'markers',
        marker = dict(
            size = 35,
            color = 'rgba(50, 205, 50, .1)'
        ),
        name = 'MaxRel60'
    )

    ema5 = go.Scatter(
        x = df.index,
        y = df['EMA_5'],
        mode = 'lines',
        line = dict(color='blue'),
        name = 'EMA5'
    )

    ema20 = go.Scatter(
        x = df.index,
        y = df['EMA_20'],
        mode = 'lines',
        line = dict(color='limegreen'),
        name = 'EMA20'
    )

    ema50 = go.Scatter(
        x = df.index,
        y = df['EMA_50'],
        mode = 'lines',
        line = dict(color='orange'),
        name = 'EMA50'
    )
    
    ema100 = go.Scatter(
        x = df.index,
        y = df['EMA_100'],
        mode = 'lines',
        line = dict(color='red'),
        name = 'EMA100'
    )
    
    psar = go.Scatter(
        x = df.index,
        y = df['PSAR'], 
        mode = 'markers',
        marker = dict(
            size = 2, 
            color = 'rgba(0, 0, 0, .8)',  
        ),
        name = 'SAR Parabolico'
    )
    
    stup = np.where(df['SUPERTd'] == 1, df['SUPERT'], np.nan)
    stdown = np.where(df['SUPERTd'] == -1, df['SUPERT'], np.nan)
    stupfill = np.where(df['SUPERTd'] == 1, df['SUPERT'], df['Close'])
    stdownfill = np.where(df['SUPERTd'] == -1, df['SUPERT'], df['Close'])
    
    strendup = go.Scatter(
        x = df.index,
        y = stup, 
        mode = 'lines',
        line = dict(
            color='limegreen',
            width=1
        ),
        connectgaps = False,
        name = 'SuperTrend'
    )
        
    strendupfill = go.Scatter(
        x = df.index,
        y = stupfill, 
        mode = 'lines',
        line = dict(
            color='rgba(0,0,0,0)',
            width=1
        ),
        connectgaps = False,
        fill='tonexty',   
        fillcolor='rgba(50, 205, 50, 0.1)', 
        showlegend=False,
        name = 'SuperTrend'
    )
        
    strenddown = go.Scatter(
        x = df.index,
        y = stdown, 
        mode = 'lines',
        line = dict(
            color='red',
            width=1
        ),
        connectgaps = False,
        name = 'SuperTrend'
    )
    
    strenddownfill = go.Scatter(
        x = df.index,
        y = stdownfill, 
        mode = 'lines',
        line = dict(
            color='rgba(0,0,0,0)',
            width=1
        ),
        fill='tonexty',   
        fillcolor='rgba(255, 0, 0, 0.1)', 
        connectgaps = False,
        showlegend=False,
        name = 'SuperTrend'
    )
    
    layout = dict(xaxis = dict(autorange=True),
                  yaxis = dict(title = 'Close', autorange=True),
                  autosize = True,
                  margin = go.layout.Margin(
                      l=0,  # Sinistra
                      r=0,  # Destra
                      b=0,  # Basso
                      t=50,  # Alto
                      pad=0  # Padding
                  ),
                  legend = dict(traceorder = 'normal', bordercolor = 'black')
    )
        
    fig = sp.make_subplots(rows=1, cols=1, shared_xaxes=True)
    fig.update_layout(layout)

# RIGA 1

    fig.add_trace(close, row=1, col=1)
    fig.add_trace(strendupfill, row=1, col=1)
    fig.add_trace(strendup, row=1, col=1)
    fig.add_trace(close2, row=1, col=1)
    fig.add_trace(strenddownfill, row=1, col=1)
    fig.add_trace(strenddown, row=1, col=1)
#    fig.add_trace(min5, row=1, col=1); fig.add_trace(max5, row=1, col=1)
#    fig.add_trace(min10, row=1, col=1); fig.add_trace(max10, row=1, col=1)
#    fig.add_trace(min20, row=1, col=1); fig.add_trace(max20, row=1, col=1)
#    fig.add_trace(min60, row=1, col=1); fig.add_trace(max60, row=1, col=1)
    fig.add_trace(ema5, row=1, col=1); fig.add_trace(ema20, row=1, col=1); fig.add_trace(ema50, row=1, col=1); fig.add_trace(ema100, row=1, col=1)
    fig.add_trace(psar, row=1, col=1)
    
    return fig


def grafico_indicatori(df): 
    close = go.Scatter(
        x = df.index,
        y = df['Close'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 0, .9)'),
        name = 'Close'
    )

    close2 = go.Scatter( # serve solo per il fill del supertrend
        x = df.index,
        y = df['Close'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 0, 0)'),
        showlegend=False,
        name = 'Close2'
    )

    min5 = go.Scatter(
        x = df[df['MaxMinRel'] == -5].index,
        y = df[df['MaxMinRel'] == -5]['Close'],
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(255, 0, 0, .9)'
        ),
        name = 'MinRel5'
    )
  
    max5 = go.Scatter(
        x = df[df['MaxMinRel'] == 5].index,
        y = df[df['MaxMinRel'] == 5]['Close'],
        mode = 'markers',
        marker = dict(
            size = 5,
            color = 'rgba(50, 205, 50, .9)'
        ),
        name = 'MaxRel5'
    )

    min10 = go.Scatter(
        x = df[df['MaxMinRel'] == -10].index,
        y = df[df['MaxMinRel'] == -10]['Close'],
        mode = 'markers',
        marker = dict(
            size = 15,
            color = 'rgba(255, 0, 0, .4)'
        ),
        name = 'MinRel10'
    )
  
    max10 = go.Scatter(
        x = df[df['MaxMinRel'] == 10].index,
        y = df[df['MaxMinRel'] == 10]['Close'],
        mode = 'markers',
        marker = dict(
            size = 15,
            color = 'rgba(50, 205, 50, .4)'
        ),
        name = 'MaxRel10'
    )

    min20 = go.Scatter(
        x = df[df['MaxMinRel'] == -20].index,
        y = df[df['MaxMinRel'] == -20]['Close'],
        mode = 'markers',
        marker = dict(
            size = 25,
            color = 'rgba(255, 0, 0, .2)'
        ),
        name = 'MinRel20'
    )
  
    max20 = go.Scatter(
        x = df[df['MaxMinRel'] == 20].index,
        y = df[df['MaxMinRel'] == 20]['Close'],
        mode = 'markers',
        marker = dict(
            size = 25,
            color = 'rgba(50, 205, 50, .2)'
        ),
        name = 'MaxRel20'
    )

    min60 = go.Scatter(
        x = df[df['MaxMinRel'] == -60].index,
        y = df[df['MaxMinRel'] == -60]['Close'],
        mode = 'markers',
        marker = dict(
            size = 35,
            color = 'rgba(255, 0, 0, .1)'
        ),
        name = 'MinRel60'
    )
  
    max60 = go.Scatter(
        x = df[df['MaxMinRel'] == 60].index,
        y = df[df['MaxMinRel'] == 60]['Close'],
        mode = 'markers',
        marker = dict(
            size = 35,
            color = 'rgba(50, 205, 50, .1)'
        ),
        name = 'MaxRel60'
    )

    ema5 = go.Scatter(
        x = df.index,
        y = df['EMA_5'],
        mode = 'lines',
        line = dict(color='blue'),
        name = 'EMA5'
    )

    ema20 = go.Scatter(
        x = df.index,
        y = df['EMA_20'],
        mode = 'lines',
        line = dict(color='limegreen'),
        name = 'EMA20'
    )

    ema50 = go.Scatter(
        x = df.index,
        y = df['EMA_50'],
        mode = 'lines',
        line = dict(color='orange'),
        name = 'EMA50'
    )
    
    ema100 = go.Scatter(
        x = df.index,
        y = df['EMA_100'],
        mode = 'lines',
        line = dict(color='red'),
        name = 'EMA100'
    )
    
    macd = go.Scatter(
        x = df.index,
        y = df['MACD'],
        mode = 'lines',
        line = dict(color='blue', width=1),
        name = 'MACD'
    )

    group1 = df[(df['MACDh'] >= 0) & (np.diff(df['MACDh'], prepend=0) >= 0)]['MACDh']  # positivo in aumento
    group2 = df[(df['MACDh'] >= 0) & (np.diff(df['MACDh'], prepend=0) < 0)]['MACDh']  # positivo in diminuzione
    group3 = df[(df['MACDh'] < 0) & (np.diff(df['MACDh'], prepend=0) > 0)]['MACDh']  # negativo in aumento
    group4 = df[(df['MACDh'] < 0) & (np.diff(df['MACDh'], prepend=0) <= 0)]['MACDh']  # negativo in diminuzione
    
    macd_h1 = go.Bar(
        x = group1.index,
        y = group1,
        marker_color='limegreen',
        name = 'Istogramma MACD'
    )

    macd_h2 = go.Bar(
        x = group2.index,
        y = group2,
        marker_color='lightgreen',
        name = 'Istogramma MACD'
    )

    macd_h3 = go.Bar(
        x = group3.index,
        y = group3,
        marker_color='lightsalmon',
        name = 'Istogramma MACD'
    )

    macd_h4 = go.Bar(
        x = group4.index,
        y = group4,
        marker_color='red',
        name = 'Istogramma MACD'
    )
    
    psar = go.Scatter(
        x = df.index,
        y = df['PSAR'], 
        mode = 'markers',
        marker = dict(
            size = 2, 
            color = 'rgba(0, 0, 0, .8)',  
        ),
        name = 'SAR Parabolico'
    )
    
    stup = np.where(df['SUPERTd'] == 1, df['SUPERT'], np.nan)
    stdown = np.where(df['SUPERTd'] == -1, df['SUPERT'], np.nan)
    stupfill = np.where(df['SUPERTd'] == 1, df['SUPERT'], df['Close'])
    stdownfill = np.where(df['SUPERTd'] == -1, df['SUPERT'], df['Close'])
    
    strendup = go.Scatter(
        x = df.index,
        y = stup, 
        mode = 'lines',
        line = dict(
            color='limegreen',
            width=1
        ),
        connectgaps = False,
        name = 'SuperTrend'
    )
        
    strendupfill = go.Scatter(
        x = df.index,
        y = stupfill, 
        mode = 'lines',
        line = dict(
            color='rgba(0,0,0,0)',
            width=1
        ),
        connectgaps = False,
        fill='tonexty',   
        fillcolor='rgba(50, 205, 50, 0.1)', 
        showlegend=False,
        name = 'SuperTrend'
    )
        
    strenddown = go.Scatter(
        x = df.index,
        y = stdown, 
        mode = 'lines',
        line = dict(
            color='red',
            width=1
        ),
        connectgaps = False,
        name = 'SuperTrend'
    )
    
    strenddownfill = go.Scatter(
        x = df.index,
        y = stdownfill, 
        mode = 'lines',
        line = dict(
            color='rgba(0,0,0,0)',
            width=1
        ),
        fill='tonexty',   
        fillcolor='rgba(255, 0, 0, 0.1)', 
        connectgaps = False,
        showlegend=False,
        name = 'SuperTrend'
    )
    
    dmosc = go.Scatter(
        x = df.index,
        y = df['DM_OSC'],
        mode = 'lines',
        line = dict(color='magenta', width=1),
        name = 'DM Osc.'
    )

    tsi = go.Scatter(
        x = df.index,
        y = df['TSI'],
        mode = 'lines',
        line = dict(color='red', width=1),
        name = 'TSI'
    )
        
    line1 = go.layout.Shape(
        type="line",
        x0=df.index[0],
        y0=25,
        x1=df.index[-1],
        y1=25,
        xref='x',
        yref='y',
        line=dict(
            color="gray",
            width=1,
            dash="dash"
        ),
    )
    
    line2 = go.layout.Shape(
        type="line",
        x0=df.index[0],
        y0=-25,
        x1=df.index[-1],
        y1=-25,
        xref='x',
        yref='y',
        line=dict(
            color="gray",
            width=1,
            dash="dash"
        ),
    )
    
    area_fill = go.layout.Shape(
        type="rect",
        x0=df.index[0],
        y0=25,
        x1=df.index[-1],
        y1=-25,
        xref='x',
        yref='y',
        fillcolor="LightSkyBlue",
        opacity=0.2,
        line_width=0,
    )

    atr = go.Scatter(
        x = df.index,
        y = df['ATR'],
        mode = 'lines',
        line = dict(color='brown', width=1),
        name = 'ATR'
    )

    layout = dict(xaxis = dict(domain=[0, 0.49], autorange=True),
                  xaxis2 = dict(domain=[0.51, 1], matches='x', autorange=True),
                  xaxis3 = dict(domain=[0, 0.49], matches='x', autorange=True),
                  xaxis4 = dict(domain=[0.51, 1], matches='x', autorange=True),
                  xaxis5 = dict(domain=[0, 0.49], matches='x', autorange=True),
                  xaxis6 = dict(domain=[0.51, 1], matches='x', autorange=True),
                  yaxis = dict(title = 'Close', autorange=True, domain=[0.61, 1]),
                  yaxis2 = dict(title = 'DM_OSC', autorange=True, domain=[0.61, 1], zeroline=True, zerolinewidth=1, zerolinecolor="black"),
                  yaxis3 = dict(title = 'MACD', autorange=True, domain=[0.31, 0.59], zeroline=True, zerolinewidth=1, zerolinecolor="black"),
                  yaxis4 = dict(title = 'TSI', autorange=True, domain=[0.31, 0.59], zeroline=True, zerolinewidth=1, zerolinecolor="black"),
                  yaxis5 = dict(title = 'MACD Hist.', autorange=True, domain=[0, 0.29], zeroline=True, zerolinewidth=1, zerolinecolor="black"),
                  yaxis6 = dict(title = 'ATR', autorange=True, domain=[0, 0.29]),
                  barmode = 'overlay',
                  autosize = True,
                  margin = go.layout.Margin(
                      l=0,  # Sinistra
                      r=0,  # Destra
                      b=0,  # Basso
                      t=50,  # Alto
                      pad=0  # Padding
                  ),
                  legend = dict(traceorder = 'normal', bordercolor = 'black')
    )
        
    fig = sp.make_subplots(rows=3, cols=2, shared_xaxes=True)
    fig.update_layout(layout)

# RIGA 1

    fig.add_trace(close, row=1, col=1)
    fig.add_trace(strendupfill, row=1, col=1)
    fig.add_trace(strendup, row=1, col=1)
    fig.add_trace(close2, row=1, col=1)
    fig.add_trace(strenddownfill, row=1, col=1)
    fig.add_trace(strenddown, row=1, col=1)
#    fig.add_trace(min5, row=1, col=1); fig.add_trace(max5, row=1, col=1)
#    fig.add_trace(min10, row=1, col=1); fig.add_trace(max10, row=1, col=1)
    fig.add_trace(min20, row=1, col=1); fig.add_trace(max20, row=1, col=1)
#    fig.add_trace(min60, row=1, col=1); fig.add_trace(max60, row=1, col=1)
    fig.add_trace(ema5, row=1, col=1); fig.add_trace(ema20, row=1, col=1); fig.add_trace(ema50, row=1, col=1); fig.add_trace(ema100, row=1, col=1)
    fig.add_trace(psar, row=1, col=1)
    
    fig.add_trace(dmosc, row=1, col=2)

# RIGA 2

    fig.add_trace(macd, row=2, col=1)

    fig.add_trace(tsi, row=2, col=2)
    fig.add_shape(area_fill, row=2, col=2)
    fig.add_shape(line1, row=2, col=2)
    fig.add_shape(line2, row=2, col=2)

# RIGA 3

    fig.add_trace(macd_h1, row=3, col=1)
    fig.add_trace(macd_h2, row=3, col=1)
    fig.add_trace(macd_h3, row=3, col=1)
    fig.add_trace(macd_h4, row=3, col=1)    

    fig.add_trace(atr, row=3, col=2)
    
    return fig
