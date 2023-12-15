
import pandas as pd
import numpy as np
import yfinance as yf
from multiprocessing import Pool, cpu_count, Manager, Value
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import Bidirectional, BatchNormalization, LSTM, Dropout, Dense, Conv1D, Flatten, GRU, Attention, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import regularizers
from tensorflow.keras.metrics import Precision, Recall, AUC
from datetime import timedelta
import json
import os
from functools import partial
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import pandas_ta as ta
import random
import h5py
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.subplots as sp
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

_features_scala_prezzo_tutte = [
    "Close",
    "EMA_5", 
    "EMA_20", 
    "EMA_50",
    "EMA_100",
    "Open",  
    "High",
    "Low",
    "PSAR",
    "SUPERT"
]

_features_da_scalare_singolarmente_tutte = [
    "Volume",
    "ATR",
    "PSARaf",
    "ADX",
    "OBV"
]

_features_oscillatori_tutte = [
    "MACDh",    
    "MACD",
    "MACDs",
    "AROONOSC",
    "TRIX",
    "TRIXs",
    "DM_OSC",
    "TSI",
    "TSIs",
    "ROC_10",
    "KVO",
    "KVOs",
    "VI_OSC"
]

_features_no_scala_tutte = [
    "SUPERTd",  
    "PSARr",
    "CMF",
    "VHF",
    "VTX_OSC"
]

_features_candele_tutte = [
    "CDL_2CROWS", "CDL_3BLACKCROWS", "CDL_3INSIDE", "CDL_3LINESTRIKE", "CDL_3OUTSIDE", "CDL_3STARSINSOUTH", "CDL_3WHITESOLDIERS", "CDL_ABANDONEDBABY", "CDL_ADVANCEBLOCK", "CDL_BELTHOLD", "CDL_BREAKAWAY", "CDL_CLOSINGMARUBOZU", "CDL_CONCEALBABYSWALL", "CDL_COUNTERATTACK", "CDL_DARKCLOUDCOVER", "CDL_DOJI_10_0.1", "CDL_DOJISTAR", "CDL_DRAGONFLYDOJI", "CDL_ENGULFING", "CDL_EVENINGDOJISTAR", "CDL_EVENINGSTAR", "CDL_GAPSIDESIDEWHITE", "CDL_GRAVESTONEDOJI", "CDL_HAMMER", "CDL_HANGINGMAN", "CDL_HARAMI", "CDL_HARAMICROSS", "CDL_HIGHWAVE", "CDL_HIKKAKE", "CDL_HIKKAKEMOD", "CDL_HOMINGPIGEON", "CDL_IDENTICAL3CROWS", "CDL_INNECK", "CDL_INSIDE", "CDL_INVERTEDHAMMER", "CDL_KICKING", "CDL_KICKINGBYLENGTH", "CDL_LADDERBOTTOM", "CDL_LONGLEGGEDDOJI", "CDL_LONGLINE", "CDL_MARUBOZU", "CDL_MATCHINGLOW", "CDL_MATHOLD", "CDL_MORNINGDOJISTAR", "CDL_MORNINGSTAR", "CDL_ONNECK", "CDL_PIERCING", "CDL_RICKSHAWMAN", "CDL_RISEFALL3METHODS", "CDL_SEPARATINGLINES", "CDL_SHOOTINGSTAR", "CDL_SHORTLINE", "CDL_SPINNINGTOP", "CDL_STALLEDPATTERN", "CDL_STICKSANDWICH", "CDL_TAKURI", "CDL_TASUKIGAP", "CDL_THRUSTING", "CDL_TRISTAR", "CDL_UNIQUE3RIVER", "CDL_UPSIDEGAP2CROWS", "CDL_XSIDEGAP3METHODS",
]

def inizializza_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print("nessuna GPU")
    
def pct_change(valore_iniziale, valore_finale):
    try:
        return ((valore_finale - valore_iniziale) / valore_iniziale) * 100
    except ZeroDivisionError:
        return 0

def analizza_ticker(nome_simbolo, start, end, progress=True, dropna_iniziali=False, dropna_finali=False):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    df = yf.download(nome_simbolo, start=start_str, end=end_str, progress=progress)
    df.index = pd.to_datetime(df.index)
    df = crea_indicatori(df)
    if dropna_iniziali:
        idx = df[df.notna().all(axis=1) == True].index[0]
        df = df[idx:]
    if dropna_finali:
        idx = df[df.notna().all(axis=1) == True].index[-1]
        df = df[:idx]
    return df

def dropna_iniziali(df):
    idx = df[df.notna().all(axis=1) == True].index[0]
    df = df[idx:]
    return df

def dropna_finali(df):
    idx = df[df.notna().all(axis=1) == True].index[-1]
    df = df[:idx]
    return df

def imposta_target(df):
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
    # df = __calcolo_drawdown_gain(df, 20)
    # df = __calcolo_drawdown_gain(df, 50)
    # df = __calcolo_drawdown_gain(df, 100)
    # df["max_gain"] = df[["Perc_Max_High_Futuro_20d", "Perc_Max_High_Futuro_50d", "Perc_Max_High_Futuro_100d"]].max(axis=1)
    # df["max_drawdown"] = df[["Perc_Drawdown_20d", "Perc_Drawdown_50d", "Perc_Drawdown_100d"]].min(axis=1)

    df['EMA_20_5d'] = df['EMA_20'].shift(-5)
    df['EMA_20_10d'] = df['EMA_20'].shift(-10)
    df['EMA_20_15d'] = df['EMA_20'].shift(-15)
    df['EMA_20_20d'] = df['EMA_20'].shift(-20)
    
    df['EMA_50_5d'] = df['EMA_50'].shift(-5)
    df['EMA_50_10d'] = df['EMA_50'].shift(-10)
    df['EMA_50_15d'] = df['EMA_50'].shift(-15)
    df['EMA_50_20d'] = df['EMA_50'].shift(-20)
    
    df['Close_5d'] = df['Close'].shift(-5)
    df['Close_10d'] = df['Close'].shift(-10)
    df['Close_15d'] = df['Close'].shift(-15)
    df['Close_20d'] = df['Close'].shift(-20)
    
    df['EMA_5_5d'] = df['EMA_5'].shift(-5)
    df['EMA_5_10d'] = df['EMA_5'].shift(-10)
    df['EMA_5_15d'] = df['EMA_5'].shift(-15)
    df['EMA_5_20d'] = df['EMA_5'].shift(-20)
    #df['Close_1d'] = df['Close'].shift(-1)
    #df['perc_EMA_5_20d'] = ((df['EMA_5_20d'] - df['EMA_5']) / df['EMA_5']) * 100
    #df['perc_Close_20d'] = ((df['Close_20d'] - df['Close']) / df['Close']) * 100
    #df['incrocio_verde_gialla'] = (ta.cross(df['EMA_20'], df['EMA_50'], above=True)).astype("int8")
    #df["incrocio_passato_verde_gialla_10d"] = df["incrocio_verde_gialla"].rolling(10).sum()
    df['Max_Close_20d'] = df['Close'].shift(-20).rolling(window=20, min_periods=1).max()
    df['pct_change_20d'] = df.apply(lambda row: pct_change(row['Close'], row['Max_Close_20d']), axis=1)
    df.drop(columns=["Max_Close_20d"], inplace=True, axis=1)

    # df["MaxMinRel"] = 0
    # df = __trova_massimi_minimi(df, 20)   
    # df = __trova_massimi_minimi(df, 50)   
    # df = __trova_massimi_minimi(df, 100)         

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df['Target_ingresso'] = (
        (df['pct_change_20d'] > 20)
    )
    df['Target_uscita'] = (
        (df['EMA_5'] > df['EMA_20']) & (df['EMA_5_5d'] < df['EMA_20_5d']) & (df['EMA_5_10d'] < df['EMA_20_10d'])
    )    
    return df
    
def grafico(df):
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

    # min5 = go.Scatter(
    #     x = df[df['MaxMinRel'] == -5].index,
    #     y = df[df['MaxMinRel'] == -5]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 5,
    #         color = 'rgba(255, 0, 0, .9)'
    #     ),
    #     name = 'MinRel5'
    # )
  
    # max5 = go.Scatter(
    #     x = df[df['MaxMinRel'] == 5].index,
    #     y = df[df['MaxMinRel'] == 5]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 5,
    #         color = 'rgba(50, 205, 50, .9)'
    #     ),
    #     name = 'MaxRel5'
    # )

    # min10 = go.Scatter(
    #     x = df[df['MaxMinRel'] == -10].index,
    #     y = df[df['MaxMinRel'] == -10]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 15,
    #         color = 'rgba(255, 0, 0, .4)'
    #     ),
    #     name = 'MinRel10'
    # )
  
    # max10 = go.Scatter(
    #     x = df[df['MaxMinRel'] == 10].index,
    #     y = df[df['MaxMinRel'] == 10]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 15,
    #         color = 'rgba(50, 205, 50, .4)'
    #     ),
    #     name = 'MaxRel10'
    # )

    # min20 = go.Scatter(
    #     x = df[df['MaxMinRel'] == -20].index,
    #     y = df[df['MaxMinRel'] == -20]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 25,
    #         color = 'rgba(255, 0, 0, .2)'
    #     ),
    #     name = 'MinRel20'
    # )
  
    # max20 = go.Scatter(
    #     x = df[df['MaxMinRel'] == 20].index,
    #     y = df[df['MaxMinRel'] == 20]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 25,
    #         color = 'rgba(50, 205, 50, .2)'
    #     ),
    #     name = 'MaxRel20'
    # )

    # min60 = go.Scatter(
    #     x = df[df['MaxMinRel'] == -60].index,
    #     y = df[df['MaxMinRel'] == -60]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 35,
    #         color = 'rgba(255, 0, 0, .1)'
    #     ),
    #     name = 'MinRel60'
    # )
  
    # max60 = go.Scatter(
    #     x = df[df['MaxMinRel'] == 60].index,
    #     y = df[df['MaxMinRel'] == 60]['Close'],
    #     mode = 'markers',
    #     marker = dict(
    #         size = 35,
    #         color = 'rgba(50, 205, 50, .1)'
    #     ),
    #     name = 'MaxRel60'
    # )

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
        visible = False,
        showlegend=True,
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
        visible = False,
        showlegend=True,
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
        visible = False,
        showlegend=True,
        fill='tonexty',   
        fillcolor='rgba(50, 205, 50, 0.1)', 
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
        visible = False,
        showlegend=True,
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
        visible = False,
        showlegend=True,
        name = 'SuperTrend'
    )
    
    target_in = go.Scatter(
        x = df[df['Target_ingresso'] == 1].index,
        y = df[df['Target_ingresso'] == 1]['Close'],
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(0, 200, 0, .9)'
        ),
        name = 'target_in'
    )
    
    target_out = go.Scatter(
        x = df[df['Target_uscita'] == 1].index,
        y = df[df['Target_uscita'] == 1]['Close'],
        mode = 'markers',
        marker = dict(
            size = 10,
            color = 'rgba(220, 0, 0, .9)'
        ),
        name = 'target_out'
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
    #fig.add_trace(min5, row=1, col=1); fig.add_trace(max5, row=1, col=1)
    #fig.add_trace(min10, row=1, col=1); fig.add_trace(max10, row=1, col=1)
    #fig.add_trace(min20, row=1, col=1); fig.add_trace(max20, row=1, col=1)
    #fig.add_trace(min60, row=1, col=1); fig.add_trace(max60, row=1, col=1)
    fig.add_trace(target_in, row=1, col=1); fig.add_trace(target_out, row=1, col=1)
    fig.add_trace(ema5, row=1, col=1); fig.add_trace(ema20, row=1, col=1); fig.add_trace(ema50, row=1, col=1); fig.add_trace(ema100, row=1, col=1)
    fig.add_trace(psar, row=1, col=1)
    
    pyo.plot(fig, filename="grafico_target.html", auto_open=True)
    
    return fig
    
def crea_indicatori(df):
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
            'KVO_34_55_13': 'KVO',
            'KVOs_34_55_13': 'KVOs',
            'DCL_20_20': 'DCL',
            'DCM_20_20': 'DCM',
            'DCU_20_20': 'DCU',
            'VTXP_20': 'VTXP',
            'VTXM_20': 'VTXM',
            'AROOND_20': 'AROOND',
            'AROONU_20': 'AROONU',
            'AROONOSC_20': 'AROONOSC',
            'NVI_1': 'NVI',
            'PVI_1': 'PVI',
            'VHF_20': 'VHF',
            'ATRr_14': 'ATR'
        })
        return df
    
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
    roc = ta.roc(close=df["Close"], length=10)
    cmf = ta.cmf(high=df["High"], low=df["Low"], close=df["Close"], volume=df['Volume'], length=10)
    trix = ta.trix(close=df['Close'], length=18)
    klinger = ta.kvo(high=df["High"], low=df["Low"], close=df["Close"], volume=df['Volume'], short=34, long=55)
    vi = ta.vortex(high=df['High'], low=df['Low'], close=df['Close'], length=20)
    aroon = ta.aroon(high=df['High'], low=df['Low'], close=df['Close'], length=20)
    nvi = ta.nvi(close=df['Close'], volume=df['Volume'])
    pvi = ta.pvi(close=df['Close'], volume=df['Volume'])
    vhf = ta.vhf(close=df['Close'], length=20)
    atr = ta.atr(high=df['High'], low=df['Low'], close=df['Close'])
    obv = ta.obv(close=df["Close"], volume=df["Volume"])
    #candele = ta.cdl_pattern(open_=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])

    df = pd.concat([df, ema5, ema20, ema50, ema100, psar, macd, tsi, supertrend, adx, trix, vi, aroon, nvi, pvi, atr, cmf, roc, klinger, vhf, obv], axis=1)

    df = __rinomina_colonne(df)
    
    #df['HLC3'] = ((df['High'] + df['Low'] + df['Close']) / 3)
    df["DM_OSC"] = (df["DMP"] - df["DMN"])
    df["VTX_OSC"] = (df["VTXP"] - df["VTXM"])
    df["VI_OSC"] = (df["PVI"] - df["NVI"])
    
    df.drop(columns=["DMP", "DMN", "VTXP", "VTXM", "PVI", "NVI", "AROOND", "AROONU"], inplace=True, axis=1)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def _scarica(nome_simbolo, scarica_prima, scarica_dopo, data_inizio, data_fine, append):
    try:
        if append == True:
            ticker = pd.read_hdf(f'tickers/{nome_simbolo}.h5', "ticker")
            ti_min = ticker.index.min()
            ti_max = ticker.index.max()
            if scarica_prima:
                inizio = data_inizio - pd.Timedelta(days=365)
                fine = ti_min - pd.Timedelta(days=1)
                df_inizio = analizza_ticker(nome_simbolo, start=inizio, end=fine, progress=False, dropna_iniziali=True, dropna_finali=False)
                ticker = pd.concat([df_inizio, ticker], axis=0, ignore_index=False)
            if scarica_dopo:
                inizio = ti_max - pd.Timedelta(days=365) 
                fine = data_fine
                df_fine = analizza_ticker(nome_simbolo, start=inizio, end=fine, progress=False, dropna_iniziali=True, dropna_finali=False)
                ticker = ticker[ticker.index < df_fine.index.min()]
                ticker = pd.concat([ticker, df_fine], axis=0, ignore_index=False)
        else:
            ticker = analizza_ticker(nome_simbolo, start=data_inizio, end=data_fine, progress=False, dropna_iniziali=True, dropna_finali=False)
        return nome_simbolo, ticker, ""
    except Exception as e:
        return nome_simbolo, None, str(e)

def _callback_tickers(result, totale_processati, tot_tickers):
    nome_simbolo, ticker, error = result
    if error == "":
        ticker.to_hdf(f'tickers/{nome_simbolo}.h5', key='ticker', mode='w')
    else:
        print(f"Errore su funzione di callback per {nome_simbolo}: {error}")

    with totale_processati.get_lock(): 
        totale_processati.value += 1
    print(f"{totale_processati.value}/{tot_tickers}) Scaricato ticker {nome_simbolo}")   
    
def _carica_screener(nome_simbolo, lista_scr, lista_scr_out, prob, nome_dataset):
    try:
        df = pd.read_hdf(f'screeners/{nome_simbolo}.h5', nome_dataset)
        df.index.set_names(['Data'], inplace=True)
        df['Ticker'] = nome_simbolo
        df.set_index('Ticker', append=True, inplace=True)
        df = df.loc[df['Previsione'] > prob]
        if nome_dataset == 'screener':
            lista_scr.append(df)
        else:
            lista_scr_out.append(df)
        return nome_simbolo, ""
    except Exception as e:
        return nome_simbolo, str(e)

def _carica_screener_callback(result, totale_processati, tot_tickers):
    nome_simbolo, error = result
    if error != "":
        print(f"Errore su funzione di callback per {nome_simbolo}: {error}")

    with totale_processati.get_lock(): 
        totale_processati.value += 1
    print(f"{totale_processati.value}/{tot_tickers}) Caricato su screener ticker {nome_simbolo}")   

def to_XY(dati_ticker, timesteps, giorni_previsione, features, targets, bilanciamento=0):
    dati_ticker = imposta_target(dati_ticker)
    features_scala_prezzo = [ft for ft in _features_scala_prezzo_tutte if ft in features]
    features_da_scalare_singolarmente = [ft for ft in _features_da_scalare_singolarmente_tutte if ft in features]
    features_oscillatori = [ft for ft in _features_oscillatori_tutte if ft in features]
    features_no_scala = [ft for ft in _features_no_scala_tutte if ft in features]
    features_candele = [ft for ft in _features_candele_tutte if ft in features]

    scalers_prezzo = []
    scaler_meno_piu = MinMaxScaler(feature_range=(-1, 1))
    scaler_standard = MinMaxScaler()

    new_dates = pd.bdate_range(start=dati_ticker.index[-1] + pd.Timedelta(days=1), periods=giorni_previsione)
    df_new = pd.DataFrame(index=new_dates)
    dati_ticker = pd.concat([dati_ticker, df_new])

    ft_prezzo = dati_ticker[features_scala_prezzo]
    ft_standard = dati_ticker[features_da_scalare_singolarmente]
    ft_meno_piu = dati_ticker[features_oscillatori]
    ft_no_scala = dati_ticker[features_no_scala]
    ft_candele = dati_ticker[features_candele] 

    _targets = dati_ticker[targets]

    i_tot = len(dati_ticker) - giorni_previsione

    tot_elementi = i_tot - (timesteps-1)    
    
    X_prezzo = X_standard = X_meno_piu = X_no_scala = X_candele = None
    if len(features_scala_prezzo) > 0:
        tot_col_prezzo_x = len(ft_prezzo.columns)
        X_prezzo = np.zeros((tot_elementi, timesteps, tot_col_prezzo_x))
    if len(features_da_scalare_singolarmente) > 0:
        tot_col_standard_x = len(ft_standard.columns)
        X_standard = np.zeros((tot_elementi, timesteps, tot_col_standard_x))
    if len(features_oscillatori) > 0:
        tot_col_meno_piu_x = len(ft_meno_piu.columns)
        X_meno_piu = np.zeros((tot_elementi, timesteps, tot_col_meno_piu_x))
    if len(features_no_scala) > 0:
        tot_col_no_scala_x = len(ft_no_scala.columns)
        X_no_scala = np.zeros((tot_elementi, timesteps, tot_col_no_scala_x))
    if len(features_candele) > 0:
        tot_col_candele_x = len(ft_candele.columns)
        X_candele = np.zeros((tot_elementi, timesteps, tot_col_candele_x))
    if len(targets) > 0:
        #tot_col_targets_y = len(targets.columns)
        #Y = np.zeros((tot_elementi, giorni_previsione, tot_col_targets_y)) # togliere se classificazione binaria
        Y = np.zeros(tot_elementi) #solo per classificazione binaria
    
    for i in range(timesteps - 1, i_tot):
        if len(features_scala_prezzo) > 0:
            arr_x = np.array(ft_prezzo.iloc[i - (timesteps - 1):i + 1])
            arr_res = arr_x.reshape(-1, 1)
            scaler_prezzo = MinMaxScaler()
            scaler_prezzo.fit(arr_res)
            arr_sc = scaler_prezzo.transform(arr_res).reshape(timesteps, tot_col_prezzo_x)
            X_prezzo[i - (timesteps - 1)] = arr_sc
            scalers_prezzo.append(scaler_prezzo)

        if len(features_da_scalare_singolarmente) > 0:
            arr_x = np.array(ft_standard.iloc[i - (timesteps - 1):i + 1])
            arr_sc = scaler_standard.fit_transform(arr_x)   
            X_standard[i - (timesteps - 1)] = arr_sc

        if len(features_oscillatori) > 0:
            arr_x = np.array(ft_meno_piu.iloc[i - (timesteps - 1):i + 1])
            arr_sc = scaler_meno_piu.fit_transform(arr_x)   
            X_meno_piu[i - (timesteps - 1)] = arr_sc

        if len(features_no_scala) > 0:
            arr_x = np.array(ft_no_scala.iloc[i - (timesteps - 1):i + 1])
            X_no_scala[i - (timesteps - 1)] = arr_x

        if len(features_candele) > 0:
            arr_x = np.array(ft_candele.iloc[i - (timesteps - 1):i + 1])
            arr_sc = scaler_meno_piu.fit_transform(arr_x) 
            X_candele[i - (timesteps - 1)] = arr_sc

        if len(targets) > 0:
            # arr_y = np.array(targets.iloc[i + 1:i + 1 + giorni_previsione]) # togliere in caso di classificazione binaria
            # arr_res = arr_y.reshape(-1, 1) # togliere in caso di classificazione binaria
            # arr_sc = scaler_prezzo.transform(arr_res).reshape(giorni_previsione, tot_col_targets_y) # togliere in caso di classificazione binaria
            # Y[i - (timesteps - 1)] = arr_sc  # togliere in caso di classificazione binaria
            Y[i - (timesteps - 1)] = np.array(_targets.iloc[i]) #solo per classificazione binaria

    X_list = [x for x in [X_prezzo, X_standard, X_meno_piu, X_no_scala, X_candele] if x is not None and x.size > 0]
    X = np.concatenate(X_list, axis=2) if X_list else np.array([])
    idx = dati_ticker.index[timesteps - 1:i_tot]

    if bilanciamento > 0:
        #rus = RandomUnderSampler(sampling_strategy=bilanciamento)
        smote = SMOTE(sampling_strategy=bilanciamento)
        dim1 = X.shape[1]
        dim2 = X.shape[2]
        X_flat = X.reshape(-1, dim1 * dim2)
        # Applica l'undersampling
        X_flat_resampled, Y = smote.fit_resample(X_flat, Y)

        # Ridimensiona X tornando alla forma originale
        X = X_flat_resampled.reshape(-1, dim1, dim2)

        # Ottieni gli indici originali dopo l'undersampling
        #idx_resampled = rus.sample_indices_

        # Usa idx_resampled per ottenere gli indici originali
        #idx = idx[idx_resampled]

    Y = Y.reshape(-1, 1)
    return idx, X, Y, scalers_prezzo

def concatena(array_list, hdf5_file, dataset_name):
    """
    Concatena una lista di array NumPy a un dataset in un file HDF5, creando il file se non esiste.
    
    :param array_list: Lista di array NumPy da aggiungere.
    :param hdf5_file: Percorso del file HDF5.
    :param dataset_name: Nome del dataset all'interno del file HDF5.
    """
    # Apri o crea il file HDF5
    with h5py.File(hdf5_file, 'a') as h5f:  # 'a' apre il file in modalità read/write e lo crea se non esiste
        # Controlla se il dataset esiste già
        if dataset_name in h5f:
            # Il dataset esiste, leggi la sua lunghezza e aggiungi gli array
            dset = h5f[dataset_name]
        else:
            # Il dataset non esiste, quindi dobbiamo crearlo
            # Usiamo la forma del primo array per definire la forma del dataset
            initial_shape = (0,) + array_list[0].shape[1:]
            maxshape = (None,) + array_list[0].shape[1:]
            
            # Crea il dataset con shape iniziale e maxshape
            dset = h5f.create_dataset(dataset_name, shape=initial_shape, maxshape=maxshape, chunks=True)
            
        # Itera su tutti gli array nella lista
        for array in array_list:
            # Calcola la nuova lunghezza del dataset
            new_len = dset.shape[0] + array.shape[0]
            # Ridimensiona il dataset per accogliere i nuovi dati
            dset.resize(new_len, axis=0)
            # Aggiungi il nuovo array alla fine del dataset
            dset[-array.shape[0]:] = array

class Posizione:
    def __init__(self, simbolo, dati_ticker, data, prezzo, n_azioni, stop_loss=None, take_profit=None):
        self.simbolo = simbolo
        self.ticker = dati_ticker.copy()
        self.data_apertura = data
        self.data_corrente = data
        self.data_minimo = data
        self.data_massimo = data
        self.data_chiusura = None
        self.prezzo_apertura = prezzo
        self.prezzo_precedente = self.prezzo_apertura
        self.prezzo_corrente = self.prezzo_apertura
        self.prezzo_minimo = self.prezzo_apertura
        self.prezzo_massimo = self.prezzo_apertura
        self.prezzo_chiusura = None
        self.pct_change = 0
        self.pct_min = 0
        self.pct_max = 0
        self.n_azioni = n_azioni
        self.giorni_apertura = 1
        self.stato = 'APERTURA'
        
        if stop_loss is None:
            self.stop_loss = 0
        else:
            self.stop_loss = prezzo * (1 - stop_loss)
            
        if take_profit is None:
            self.take_profit = np.inf
        else:
            self.take_profit = prezzo * (1 + take_profit)
    
    def __str__(self):
        return self.simbolo
    
    def to_dict(self):
        dict = {
            'Simbolo': self.simbolo,
            'Data_apertura': self.data_apertura,
            'Prezzo_apertura': self.prezzo_apertura,
            'Data_corrente': self.data_corrente, 
            'Prezzo_corrente': self.prezzo_corrente,
            'Data_minimo': self.data_minimo, 
            'Prezzo_minimo': self.prezzo_minimo,
            'Data_massimo': self.data_massimo, 
            'Prezzo_massimo': self.prezzo_massimo,
            'Data_chiusura': self.data_chiusura,
            'Prezzo_chiusura': self.prezzo_chiusura,
            'Guadagno_corrente': self.pct_change,
            'Guadagno_minimo': self.pct_min,
            'Guadagno_massimo': self.pct_max,
            'Giorni_apertura': self.giorni_apertura, 
            'Stato': self.stato
        }
        return dict

    def step(self):
        self.data_corrente += timedelta(days=1)
        if self.data_corrente in self.ticker.index:
            prezzi = self.ticker.loc[self.data_corrente]
            low = prezzi['Low']
            high = prezzi['High']
            open = prezzi['Open']
            close = prezzi['Close']
            if low < self.stop_loss:
                self.chiudi(self.data_corrente, self.stop_loss)
                self.stato = 'CHIUSURA su STOP LOSS'
            elif high > self.take_profit:
                self.chiudi(self.data_corrente, self.take_profit)
                self.stato = 'CHIUSURA su TAKE PROFIT'
            else:
                self.pct_change = pct_change(self.prezzo_precedente, close)
                self.prezzo_precedente = close
                self.prezzo_corrente = close
                self.stato = 'in corso'
                if high > self.prezzo_massimo:
                    self.prezzo_massimo = high
                    self.data_massimo = self.data_corrente
                    self.pct_max = pct_change(self.prezzo_apertura, high)
                if low < self.prezzo_minimo:
                    self.prezzo_minimo = low  
                    self.data_minimo = self.data_corrente  
                    self.pct_min = pct_change(self.prezzo_apertura, low)
                self.giorni_apertura += 1

    def chiudi(self):
        prezzo = self.ticker.loc[self.data_corrente, 'Open']
        self.prezzo_corrente = prezzo
        self.prezzo_chiusura = prezzo
        self.stato = 'CHIUSURA'
        self.data_chiusura = self.data_corrente

class Borsa:
    def __init__(self, n_simboli_contemporanei=10, bilancio_iniziale=1000, probabilità_per_acquisto=0.5, giorni_max_posizione=20, stop_loss=None, take_profit=None, data_inizio=pd.Timestamp(year=2005, month=1, day=1), data_fine=pd.Timestamp.now().normalize()):
        self.N_SIMBOLI = n_simboli_contemporanei
        self.DATA_INIZIO = pd.to_datetime(data_inizio)
        self.DATA_FINE = pd.to_datetime(data_fine)
        self.BILANCIO_INIZIALE = bilancio_iniziale
        self.PROBABILITA_PER_ACQUISTO = probabilità_per_acquisto
        self.SL = stop_loss
        self.TP = take_profit
        self.GIORNI_POS = giorni_max_posizione
        self._posizioni = []
        self._valore_posizioni = 0
        self._bilancio = None
        self._data_corrente = None
        self._bilancio_per_simbolo = None
        self.esito_trading = None

        self.modello_ingresso = Modello()
        self.modello_ingresso.carica(progetto='mod_1_in')
        self.timesteps = self.modello_ingresso.timesteps
        self.giorni_previsione = self.modello_ingresso.giorni_previsione
        self.features = self.modello_ingresso.features
        self.targets = self.modello_ingresso.targets
        
        self.modello_uscita = Modello()
        self.modello_uscita.carica(progetto='mod_1_out')
        self.features_out = self.modello_uscita.features
        self.targets_out = self.modello_uscita.targets        

        self.lista_tickers = pd.read_parquet("lista_ticker.parquet")
        self.tot_tickers = len(self.lista_tickers)
        self.screener = pd.DataFrame()
        self.screener_out = pd.DataFrame()

    def aggiorna_dati(self):
        try:
            if os.path.exists(f'_indice.json'):
                with open(f'_indice.json', 'r') as jsonfile:
                    indice = json.load(jsonfile)
                prima_data = pd.to_datetime(indice['prima_data'])
                ultima_data = pd.to_datetime(indice['ultima_data'])
                
                if (self.DATA_INIZIO < prima_data):
                    scarica_prima = True
                else:
                    scarica_prima = False

                if (self.DATA_FINE > ultima_data):
                    scarica_dopo = True
                else:
                    scarica_dopo = False

                if scarica_prima or scarica_dopo:
                    print('Caricamento nuovi dati ticker')
                    self.scarica_tickers(scarica_prima, scarica_dopo, self.DATA_INIZIO, self.DATA_FINE, append=True)
                    print('Aggiornamento lista tickers')  
                    self.aggiorna_lista_tickers()      
                    print('Aggiornamento screener')
                    self.avvia_screener(append=True)    
                else:
                    print('Caricamento screener esistenti')
                    self.screener = pd.read_hdf('screeners/_screener.h5', 'screener')
                    self.screener_out = pd.read_hdf('screeners/_screener.h5', 'screener_out')
            else:
                print('Scarico totale dati ticker')
                self.scarica_tickers(scarica_prima=True, scarica_dopo=True, data_inizio=self.DATA_INIZIO, data_fine=self.DATA_FINE, append=False)
                print('Creazione nuovi screener')
                self.avvia_screener(append=False)  
                print('Aggiornamento lista tickers')  
                self.aggiorna_lista_tickers()              
        except Exception as e:
            print(str(e))
        
    def scarica_tickers(self, scarica_prima=True, scarica_dopo=True, data_inizio=pd.Timestamp(year=2005, month=1, day=1), data_fine=pd.Timestamp.now().normalize(), append=False) -> None: 
        totale_processati = Value('i', 0)
        with Pool(cpu_count()) as p:
            callback_with_args = partial(_callback_tickers, totale_processati=totale_processati, tot_tickers=self.tot_tickers)
            for i in range(0, self.tot_tickers):
                nome_simbolo = self.lista_tickers.iloc[i]["Ticker"]
                param = (nome_simbolo, scarica_prima, scarica_dopo, data_inizio, data_fine, append)
                p.apply_async(_scarica, args=param, callback=callback_with_args)
            p.close()
            p.join()     

        indice = {
            'prima_data': data_inizio.strftime('%Y-%m-%d'),
            'ultima_data': data_fine.strftime('%Y-%m-%d')
        }
        with open(f'_indice.json', 'w') as jsonfile:
            json.dump(indice, jsonfile, indent=4)    

    def aggiorna_lista_tickers(self):
        cartella_tickers = 'tickers'
        cartella_screeners = 'screeners'
        files_tickers = os.listdir(cartella_tickers)

        tickers_da_rimuovere = []

        for file in files_tickers:
            percorso_file = os.path.join(cartella_tickers, file)

            # Controlla il numero di righe nel file HDF5
            try:
                df_temp = pd.read_hdf(percorso_file)
                if len(df_temp) < 100:
                    # Elimina il file da tickers
                    os.remove(percorso_file)

                    # Elimina il corrispondente file in screeners
                    percorso_file_screener = os.path.join(cartella_screeners, file)
                    if os.path.exists(percorso_file_screener):
                        os.remove(percorso_file_screener)

                    # Aggiungi il nome del ticker alla lista dei tickers da rimuovere
                    nome_ticker = os.path.splitext(file)[0]
                    tickers_da_rimuovere.append(nome_ticker)
            except Exception as e:
                print(f"Errore nella lettura del file {file}: {e}")

        # Rimuovi i tickers dalla lista
        self.lista_tickers = self.lista_tickers[~self.lista_tickers['Ticker'].isin(tickers_da_rimuovere)]

        # Aggiorna la lista dei tickers nel DataFrame
        lista_files_aggiornata = os.listdir(cartella_tickers)
        lista_files_aggiornata = [os.path.splitext(file)[0] for file in lista_files_aggiornata]
        self.lista_tickers = self.lista_tickers[self.lista_tickers['Ticker'].isin(lista_files_aggiornata)]

        # Salva il DataFrame aggiornato
        self.lista_tickers.to_parquet('lista_ticker.parquet')
       
    def avvia_screener(self, append=False, inizia_da=0) -> None:
        tot_tickers = len(self.lista_tickers)

        for i in range(inizia_da, tot_tickers):
            try:
                nome_simbolo = self.lista_tickers.iloc[i]["Ticker"]
                print("\033[42m" + f'{i+1}/{tot_tickers}) Calcolo screeners per {nome_simbolo}' + "\033[0m")
                ticker = pd.read_hdf(f'tickers/{nome_simbolo}.h5', 'ticker')
                if append:
                    scr = pd.read_hdf(f'screeners/{nome_simbolo}.h5', 'screener')
                    scr_out = pd.read_hdf(f'screeners/{nome_simbolo}.h5', 'screener_out')
                    inizio = scr.index.max() - pd.Timedelta(days=365)
                    ticker_analisi = ticker.loc[ticker.index >= inizio].copy()
                    if scr.index.max() == ticker.index.max():
                        scr = scr.drop(scr.index[-1]).copy()
                        scr_out = scr_out.drop(scr_out.index[-1]).copy()
                    idx, X, Y, _ = to_XY(dati_ticker=ticker_analisi, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features, targets=self.targets, bilanciamento=0)
                    idx_out, X_out, Y_out, _ = to_XY(dati_ticker=ticker_analisi, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features_out, targets=self.targets_out, bilanciamento=0)
                    print(f'Aggiornamento previsione {nome_simbolo}')
                    pred = self.modello_ingresso.model.predict(X)
                    pred_out = self.modello_uscita.model.predict(X_out)
                    scr_temp = pd.DataFrame({'Previsione': pred.flatten().round(2), 'Reale': Y.flatten()}, index=idx)
                    scr_temp = scr_temp.loc[scr_temp.index > scr.index.max()].copy()
                    scr = pd.concat([scr, scr_temp], axis=0, ignore_index=False)
                    scr_temp = pd.DataFrame({'Previsione': pred_out.flatten().round(2), 'Reale': Y_out.flatten()}, index=idx_out)
                    scr_temp = scr_temp.loc[scr_temp.index > scr_out.index.max()].copy()
                    scr_out = pd.concat([scr_out, scr_temp], axis=0, ignore_index=False)
                    print(f"Aggiornamento file di screener {nome_simbolo}.h5")
                    scr.to_hdf(f'screeners/{nome_simbolo}.h5', key='screener', mode='w')
                    scr_out.to_hdf(f'screeners/{nome_simbolo}.h5', key='screener_out', mode='a')
                else:
                    idx, X, Y, _ = to_XY(dati_ticker=ticker, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features, targets=self.targets, bilanciamento=0)
                    idx_out, X_out, Y_out, _ = to_XY(dati_ticker=ticker, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features_out, targets=self.targets_out, bilanciamento=0)
                    print(f'Previsione {nome_simbolo}')
                    pred = self.modello_ingresso.model.predict(X)
                    scr = pd.DataFrame({'Previsione': pred.flatten().round(2), 'Reale': Y.flatten()}, index=idx)
                    scr = scr.loc[scr.index >= self.DATA_INIZIO]
                    pred_out = self.modello_uscita.model.predict(X_out)
                    scr_out = pd.DataFrame({'Previsione': pred_out.flatten().round(2), 'Reale': Y_out.flatten()}, index=idx_out)
                    scr_out = scr_out.loc[scr_out.index >= self.DATA_INIZIO]
                    print(f"Salvataggio file di screener {nome_simbolo}.h5")
                    scr.to_hdf(f'screeners/{nome_simbolo}.h5', key='screener', mode='w')
                    scr_out.to_hdf(f'screeners/{nome_simbolo}.h5', key='screener_out', mode='a')
            except Exception as e:
                print(str(e))
        self.carica_screener('screener')
        self.carica_screener('screener_out')
        self.screener.to_hdf('screeners/_screener.h5', key='screener', mode='w')
        self.screener_out.to_hdf('screeners/_screener.h5', key='screener_out', mode='a')

    def carica_screener(self, nome_dataset):
        manager = Manager()
        lista_scr = manager.list()
        lista_scr_out = manager.list()
        totale_processati = Value('i', 0)
        with Pool(cpu_count()) as p:
            callback_with_args = partial(_carica_screener_callback, totale_processati=totale_processati, tot_tickers=self.tot_tickers)
            for i in range(0, self.tot_tickers):
                nome_simbolo = self.lista_tickers.iloc[i]["Ticker"]
                param = (nome_simbolo, lista_scr, lista_scr_out, self.PROBABILITA_PER_ACQUISTO, nome_dataset)
                p.apply_async(_carica_screener, args=param, callback=callback_with_args)
            p.close()
            p.join()  
        if nome_dataset == 'screener':
            self.screener = pd.concat(lista_scr, axis=0, ignore_index=False)
            self.screener = self.screener.sort_values(by=['Data', 'Previsione'], ascending=[True, False])
        else:
            self.screener_out = pd.concat(lista_scr_out, axis=0, ignore_index=False)
            self.screener_out = self.screener_out.sort_values(by=['Data', 'Previsione'], ascending=[True, False])

    def ultimo_screener(self) -> (pd.Timestamp, pd.DataFrame):
        # Ottieni l'ultimo valore dell'indice di livello 0
        ultimo_indice = self.screener.index.get_level_values(0)[-1]
        ultimo_indice_out = self.screener_out.index.get_level_values(0)[-1]
        # Usa .loc per selezionare tutte le righe con quell'indice di livello 0
        return ultimo_indice.date(), self.screener.loc[ultimo_indice], ultimo_indice_out.date(), self.screener_out.loc[ultimo_indice_out]

    def reset_trading(self) -> None:
        self._data_corrente = self.DATA_INIZIO
        self._posizioni = []
        self._bilancio = self.BILANCIO_INIZIALE
        self.esito_trading = pd.DataFrame()
        self._bilancio_per_simbolo = self.BILANCIO_INIZIALE / self.N_SIMBOLI
        self._valore_posizioni = 0

    def simulazione_trading(self) -> None:
        self.reset_trading()
        while self._data_corrente <= self.DATA_FINE:
            print("\n" + "\033[42m" + str(self._data_corrente.date()) + "\033[0m")
            self._chiudi_posizioni()
            self.aggiorna_posizioni()
            self._apri_posizioni()
            self.step()

    def step(self):
        self._data_corrente += timedelta(days=1)
        for pos in self._posizioni:
            pos.step()
        
    def aggiorna_posizioni(self):
        self._valore_posizioni = 0
        for pos in self._posizioni:
            importo = pos.prezzo_corrente * pos.n_azioni
            self._valore_posizioni += importo
            #self.aggiorna_esito(pos)

    def aggiorna_esito(self, posizione):
        dict_esito = posizione.to_dict()
        dict_esito['Bilancio'] = self._bilancio
        dict_esito['Tot_posizioni_aperte'] = len(self._posizioni)
        dict_esito['Valore_posizioni_aperte'] = self._valore_posizioni
        df_esito = pd.DataFrame(dict_esito, index=[0])
        self.esito_trading = pd.concat([self.esito_trading, df_esito], axis=0, ignore_index=True)
        print(f"\rBilancio: {dict_esito['Bilancio']}, Valore posizioni: {dict_esito['Valore_posizioni_aperte']}, n.pos: {dict_esito['Tot_posizioni_aperte']}                          ", end=' ', flush=True)
        
    def _apri_posizioni(self):
        if self._data_corrente in self.screener.index.get_level_values(0):
            scr = self.screener.loc[(self._data_corrente, slice(None)), :]
            i = 0
            while (len(self._posizioni) < self.N_SIMBOLI) and (i < len(scr)):
                simbolo = scr.index[i][1]

                ticker = pd.read_hdf(f'tickers/{simbolo}.h5', 'ticker')
                if self._data_corrente in ticker.index:
                    prezzi = ticker.loc[ticker.index == self._data_corrente]
                    prezzo = prezzi["Open"].iloc[0]
                    n_azioni = self._bilancio_per_simbolo // prezzo
                    if (prezzo < self._bilancio_per_simbolo) and (simbolo not in [x.simbolo for x in self._posizioni]):
                        pos = Posizione(simbolo, ticker, self._data_corrente, prezzo, n_azioni, self.SL, self.TP)
                        self._posizioni.append(pos)
                        importo = prezzo * n_azioni
                        commissione = self.applica_commissione(importo)
                        self._bilancio = self._bilancio - importo - commissione
                        self._valore_posizioni += importo
                        self.aggiorna_esito(pos)
                        pos_da_aprire = (self.N_SIMBOLI - len(self._posizioni))
                        if pos_da_aprire == 0:
                            self._bilancio_per_simbolo = 0.
                        else:
                            self._bilancio_per_simbolo = self._bilancio / pos_da_aprire                
                i += 1
        
    def applica_commissione(self, importo_transazione, broker='FINECO'):
        if broker == 'FINECO':
            if importo_transazione < 2.95:
                return 2.95
            elif importo_transazione > 19:
                return 19.
            else:
                return importo_transazione * 0.0019
        else:
            return 0.

    def _chiudi_posizioni(self):
        if self._data_corrente in self.screener_out.index.get_level_values('Data'):
            scr_out = self.screener_out.loc[(self._data_corrente, slice(None)), :]
            scr_out = scr_out.reset_index(level='Data', drop=True)
            i = 0
            pos_da_mantenere = []
            for pos in self._posizioni:
                if (pos.simbolo in scr_out.index):
                    pos.chiudi()
                    importo = pos.prezzo_chiusura * pos.n_azioni
                    commissione = self.applica_commissione(importo)
                    print(f'importo {importo}')
                    print(f'bilancio {self._bilancio}')
                    self._bilancio = self._bilancio + importo - commissione
                    print(f'bilancio {self._bilancio}')
                    self._valore_posizioni -= importo
                    self.aggiorna_esito(pos)
                else:
                    pos_da_mantenere.append(pos)   
                i += 1
            self._posizioni = pos_da_mantenere

    # Funzione obiettivo per l'ottimizzazione
    def funzione_obiettivo(self, N_SIMBOLI, GIORNI_POS, SL, TP):
        # Imposta i parametri
        self.N_SIMBOLI = N_SIMBOLI
        self.GIORNI_POS = GIORNI_POS
        self.SL = SL
        self.TP = TP
        print('PARAMETRI:')
        print(f'N.simboli = {N_SIMBOLI}')
        print(f'Giorni pos. = {GIORNI_POS}')
        print(f'SL = {SL}')
        print(f'TP = {TP}')
        # Esegui il trading
        self.avvia_trading()

        # Restituisce il bilancio finale in modo negativo per la minimizzazione
        return -self._bilancio

class Modello:
    def __init__(self):
        pass
        
    def _crea_modello(self):      
        # Input layer
        input_layer = Input(shape=(self.timesteps, self.n_features))

        # Convolutional layer
        conv1 = Conv1D(filters=128, kernel_size=5, activation='relu')(input_layer)
        conv1 = BatchNormalization()(conv1)

        # Continuation of the model
        lstm2 = GRU(50)(conv1)
        lstm2 = Dropout(0.5)(lstm2)

        dense2 = Dense(80, activation='relu', kernel_regularizer=regularizers.l2(0.02))(lstm2)
        dense2 = Dropout(0.5)(dense2)

        batch_norm1 = BatchNormalization()(dense2)

        dense3 = Dense(40, activation='relu', kernel_regularizer=regularizers.l2(0.02))(batch_norm1)
        dense3 = Dropout(0.5)(dense3)

        batch_norm2 = BatchNormalization()(dense3)

        # Output layer
        output_layer = Dense(1, activation='sigmoid')(batch_norm2)

        adam = Adam(learning_rate=self.learning_rate)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall(), AUC(curve='PR')])

        return model
    
    def crea(self, 
             progetto='default', 
             timesteps=120, 
             giorni_previsione=1, 
             features=["EMA_20", "EMA_50", "EMA_100", "Volume", "MACDh", "AROONOSC", "TRIX", "DM_OSC", "TSI", "KVO"], 
             targets=["Target_ingresso"], 
             n_ticker_batch=400, 
             bilanciamento=1, 
             epochs=100,
             batch_size=2052, 
             soglia=0.5, 
             class_weights={0: 3, 1: 1},
             learning_rate=0.001, 
             train_test_split=0.2
            ):
        self.progetto = progetto
        self.features = features
        self.targets = targets
        self.timesteps = timesteps # n. barre del periodo passato per la ricerca di pattern, inclusa ultima data disponibile
        self.giorni_previsione = giorni_previsione  # giorni futuri di cui effettuare la previsione
        self.features_scala_prezzo = [ft for ft in _features_scala_prezzo_tutte if ft in self.features]
        self.features_da_scalare_singolarmente = [ft for ft in _features_da_scalare_singolarmente_tutte if ft in self.features]
        self.features_oscillatori = [ft for ft in _features_oscillatori_tutte if ft in self.features]
        self.features_no_scala = [ft for ft in _features_no_scala_tutte if ft in self.features]
        self.features_candele = [ft for ft in _features_candele_tutte if ft in self.features]
        self.targets = targets
        self.n_ticker_batch = n_ticker_batch
        self.bilanciamento = bilanciamento
        self.batch_size = batch_size
        self.epochs = epochs
        self.soglia = soglia
        self.learning_rate = learning_rate
        self.train_test_split = train_test_split
        self.class_weights = class_weights
        self.n_features = len(self.features) 
        self.n_targets = len(self.targets) 
        self.model = self._crea_modello() 
        self.model_history = None

    def carica(self, progetto='default'):
        self.progetto = progetto
        percorso_file = f'{self.progetto}/impostazioni.json'
        try:
            with open(percorso_file, "r") as file:
                impostazioni = json.load(file) if os.path.getsize(percorso_file) > 0 else {}
                self.timesteps = impostazioni.get("timesteps", 120) # n. barre del periodo passato per la ricerca di pattern, inclusa ultima data disponibile
                self.giorni_previsione = impostazioni.get("giorni_previsione", 1)  # giorni futuri di cui effettuare la previsione
                self.features = impostazioni.get("features", ["EMA_20", "EMA_50", "EMA_100", "Volume", "MACDh", "AROONOSC", "TRIX", "DM_OSC", "TSI", "KVO"])
                self.features_scala_prezzo = [ft for ft in _features_scala_prezzo_tutte if ft in self.features]
                self.features_da_scalare_singolarmente = [ft for ft in _features_da_scalare_singolarmente_tutte if ft in self.features]
                self.features_oscillatori = [ft for ft in _features_oscillatori_tutte if ft in self.features]
                self.features_no_scala = [ft for ft in _features_no_scala_tutte if ft in self.features]
                self.features_candele = [ft for ft in _features_candele_tutte if ft in self.features]                
                self.targets = impostazioni.get("targets", ["Target_ingresso"])
                self.n_features = len(self.features)
                self.n_targets = len(self.targets) 
                self.n_ticker_batch = impostazioni.get("n_ticker_batch", 400)
                self.bilanciamento = impostazioni.get("bilanciamento", 1)
                self.batch_size = impostazioni.get("batch_size", 2052)
                self.epochs = impostazioni.get("epochs", 100)
                self.soglia = impostazioni.get("soglia", 0.5)
                self.learning_rate = impostazioni.get("learnig_rate", 0.001)
                self.train_test_split = impostazioni.get("train_test_split", 0.2)
                self.class_weights = impostazioni.get("class_weights", {0: 3, 1: 1})
                self.n_features = len(self.features) 
                self.n_targets = len(self.targets) 
                self.model = load_model(f"{self.progetto}/model.h5")  
                self.model_history = pd.read_hdf(f'{progetto}/model_history.h5', 'history')      
        except Exception as e:
            print(f"Errore durante il caricamento del file: {e}")

    def _salva_impostazioni(self):
        impostazioni = {
            "timesteps": self.timesteps,
            "giorni_previsione": self.giorni_previsione,
            "features": self.features,
            "targets": self.targets,
            "n_ticker_batch": self.n_ticker_batch,
            "bilanciamento": self.bilanciamento,
            "batch_size": self.batch_size,
            "epochs": self.epochs, 
            "soglia": self.soglia,
            "class_weights": self.class_weights
        }
        with open(f'{self.progetto}/impostazioni.json', "w") as file:
            json.dump(impostazioni, file, indent=4)
            
        modello = json.loads(self.model.to_json())
        with open(f'{self.progetto}/struttura.json', "w") as file:
            json.dump(modello, file, indent=4)

    def salva(self):
        os.makedirs(self.progetto, exist_ok=True)
        self._salva_impostazioni()
        self.model.save(f'{self.progetto}/model.h5')
        self.model_history.to_hdf(f'{self.progetto}/model_history.h5', key='history', mode='w')

    def genera_XY(self, lista_files, nome_file=''):
        perc_file = f'XY/XY_{nome_file}.h5'
        if not os.path.exists(perc_file):
            manager = Manager()
            listaX = manager.list()
            listaY = manager.list()
            totale_processati = Value('i', 1)  
            tot_files = len(lista_files)
            with Pool(cpu_count()) as p:
                for file_name in lista_files:
                    param = (file_name, self.timesteps, self.giorni_previsione, self.features, self.targets, self.bilanciamento)
                    p.apply_async(_process_ticker, args=param, callback=lambda result: _callback_XY(result, listaX, listaY, totale_processati, tot_files, hdf5_file=perc_file))

                p.close()
                p.join()

            if len(listaX) > 0:
                print("\033[43m" + 'Salvataggio finale su file X' + "\033[0m")
                concatena(listaX, perc_file, dataset_name='X')
                print("\033[43m" + 'Salvataggio finale su file Y' + "\033[0m")
                concatena(listaY, perc_file, dataset_name='Y')
                del listaX[:]
                del listaY[:]

    def addestra(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, verbose=1)
        #model_checkpoint = ModelCheckpoint(f'{self.progetto}/model.h5', monitor='val_precision', save_best_only=True, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

        callbacks = [early_stopping, reduce_lr]
        list_of_files = os.listdir('tickers')
        random.shuffle(list_of_files)
        train_files = list_of_files[:self.n_ticker_batch]
        val_files = list_of_files[self.n_ticker_batch:int(self.n_ticker_batch*(1+self.train_test_split))]
        
        print("\033[41m" + 'Preparazione dati di train' + "\033[0m")
        self.genera_XY(train_files, 'train')
        print("\033[41m" + 'Preparazione dati di validazione' + "\033[0m")
        self.genera_XY(val_files, 'val')
        train_generator = DataGenerator('XY/XY_train.h5', self.batch_size)
        val_generator = DataGenerator('XY/XY_val.h5', self.batch_size)    
           
        history = self.model.fit(train_generator, epochs=self.epochs, validation_data=val_generator, callbacks=callbacks, class_weight=self.class_weights, steps_per_epoch=len(train_generator), validation_steps=len(val_generator))
        self.model_history = pd.DataFrame(history.history)
        self.salva()
        self.grafico_loss(salva_su_file=True)
        self.grafico_precision(salva_su_file=True)
        df = self.test()
        return df
        
    def previsione_singola(self, nome_simbolo):
        print(f'Caricamento dati ticker {nome_simbolo}')
        ticker = pd.read_hdf(f'tickers/{nome_simbolo}.h5', 'ticker')
        print('Generazione X e Y')
        idx, X, Y, _ = to_XY(dati_ticker=ticker, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features, targets=self.targets, bilanciamento=0)
        print(f'Previsione')
        pred = self.modello_ingresso.model.predict(X)
        scr = pd.DataFrame({'Previsione': pred.flatten().round(2), 'Reale': Y.flatten()}, index=idx)
        scr = scr[scr.index >= self.DATA_INIZIO]
        print(f"Salvataggio file {nome_simbolo}.h5")
    
    def grafico_loss(self, salva_su_file=False):
        num_epochs = self.model_history.shape[0]
        plt.plot(np.arange(0, num_epochs), self.model_history['loss'], label="Training")
        plt.plot(np.arange(0, num_epochs), self.model_history['val_loss'], label="Validation")
        plt.legend()
        plt.title('Loss')
        plt.tight_layout()
        if salva_su_file:
            plt.savefig(f'{self.progetto}/grafico_loss.png')
        plt.show()        

    def grafico_precision(self, salva_su_file=False):
        num_epochs = self.model_history.shape[0]
        plt.plot(np.arange(0, num_epochs), self.model_history['precision'], label="Training")
        plt.plot(np.arange(0, num_epochs), self.model_history['val_precision'], label="Validation")
        plt.legend()
        plt.title('Precision')
        plt.tight_layout()
        if salva_su_file:
            plt.savefig(f'{self.progetto}/grafico_precision.png')
        plt.show()        

    def test(self, nome_simbolo='BTG'):
        ticker = pd.read_hdf(f'tickers/{nome_simbolo}.h5', 'ticker')
        ticker = dropna_iniziali(ticker)
        ticker = dropna_finali(ticker)
        idx, X, Y, _ = to_XY(ticker, timesteps=self.timesteps, giorni_previsione=self.giorni_previsione, features=self.features, targets=self.targets, bilanciamento=0)
        print(f'X.shape: {X.shape}')
        print(f'Y.shape: {Y.shape}')
        print(f'ticker.shape: {ticker.shape}')
        pred = self.model.predict(X, batch_size=self.batch_size, verbose=1, use_multiprocessing=True)
        pred_binary = (pred > self.soglia).astype(int)
        
        result = self.model.evaluate(X, Y, batch_size=self.batch_size, verbose=1, use_multiprocessing=True, return_dict=True)
        print(result)
        
        # Visualizza come heatmap
        matrice = confusion_matrix(Y, pred_binary)
        sns.heatmap(matrice, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Previsti')
        plt.ylabel('Reali')
        plt.savefig(f'{self.progetto}/confusion_matrix.png')
        plt.show()
        print(f'idx: {idx.shape}')
        print(f'pred: {pred.shape}')
        print(f'real: {Y.shape}')
        df = pd.DataFrame({'Prev': pred.flatten().round(2), 'Real': Y.flatten()}, index=idx)
        return df       

def _process_ticker(file_name, timesteps, giorni_previsione, features, targets, bilanciamento):
    try:
        ticker = pd.read_hdf(f'tickers/{file_name}', 'ticker')
        ticker = dropna_iniziali(ticker)
        ticker = dropna_finali(ticker)
        _, X, Y, _ = to_XY(ticker, timesteps, giorni_previsione, features, targets, bilanciamento)
        return file_name, X, Y, ""
    except Exception as e:
        return file_name, np.array([]), np.array([]), str(e)

def _callback_XY(result, listaX, listaY, totale_processati, tot_files, hdf5_file):
    nome_simbolo, X, Y, err = result
    if err == "":
        if X.shape[0] > 0 and Y.shape[0] > 0:  # Verifica se X e Y sono non vuoti
            print(f'X.shape:{X.shape}')
            print(f'Y.shape:{Y.shape}')
            listaX.append(X)
            listaY.append(Y)
            print("\033[42m" + f"{totale_processati.value}/{tot_files}) Completato ticker {nome_simbolo}" + "\033[0m")
        else:
            print("\033[43m" + f"Ticker {nome_simbolo} ignorato a causa di dati mancanti o errati." + "\033[0m")
    else:
        print(err)

    with totale_processati.get_lock(): 
        totale_processati.value += 1
        if len(listaX) >= 100:
            print("\033[43m" + 'Salvataggio su file X' + "\033[0m")
            concatena(listaX, hdf5_file, dataset_name='X')
            print("\033[43m" + 'Salvataggio su file Y' + "\033[0m")
            concatena(listaY, hdf5_file, dataset_name='Y')
            del listaX[:]
            del listaY[:]
     
def modifica_target():
    totale_processati = Value('i', 1)
    list_of_files = os.listdir('tickers')
    tot_tickers = len(list_of_files)
    with Pool(cpu_count()) as p:
        callback_with_args = partial(_callback_modifica_target, totale_processati=totale_processati, tot_tickers=tot_tickers)
        for i in range(0, tot_tickers):
            nome_file = list_of_files[i]
            param = (nome_file,)
            p.apply_async(_modifica_target, args=param, callback=callback_with_args)
        p.close()
        p.join()          
   
def _modifica_target(nome_file):
    try:
        ticker = pd.read_hdf(f'tickers/{nome_file}', 'ticker')
        if 'Target_ingresso' in ticker.columns:
            ticker.drop(['Target_ingresso'], axis=1, inplace=True)
        if 'Target_uscita' in ticker.columns:
            ticker.drop(['Target_uscita'], axis=1, inplace=True)
        ticker = imposta_target(ticker)
        return nome_file, ticker, ""
    except Exception as e:
        return nome_file, ticker, str(e)

def _callback_modifica_target(result, totale_processati, tot_tickers):
    nome_file, ticker, err = result
    if err == "":
        print(f"{totale_processati.value}/{tot_tickers}) Modificato target {nome_file}")
        ticker.to_hdf(f'tickers/{nome_file}', key='ticker', mode='w')
    else:
        print(err)

    with totale_processati.get_lock(): 
        totale_processati.value += 1
        
class DataGenerator(Sequence):
    def __init__(self, file_path, batch_size):
        self.file_path = file_path
        self.batch_size = batch_size
        # Apriamo il file in modalità lettura e salviamo i riferimenti ai dataset
        self.file = h5py.File(file_path, 'r')
        self.X = self.file['X']
        self.Y = self.file['Y']
        self.num_samples = self.X.shape[0]

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, idx):
        # Calcola gli indici per il batch corrente
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        # Legge solo i dati necessari per il batch corrente
        batch_x = self.X[start:end]
        batch_y = self.Y[start:end]

        return batch_x, batch_y

    def on_epoch_end(self):
        # Eventuali azioni alla fine di ogni epoca, se necessario
        pass

    def __del__(self):
        # Assicurati di chiudere il file quando il generatore viene distrutto
        self.file.close()

def ottimizzazione_parametri():
    PROBABILITA_PER_ACQUISTO = 0.5
    BILANCIO_INIZIALE = 1000
    GIORNI_MAX_POSIZIONE = 40
    N_SIMBOLI = 10
    DATA_INIZIO = pd.Timestamp(2013, 1, 1)
    
    inizializza_gpu()

    borsa = Borsa(N_SIMBOLI, BILANCIO_INIZIALE, PROBABILITA_PER_ACQUISTO, GIORNI_MAX_POSIZIONE, data_inizio=DATA_INIZIO)
    borsa.aggiorna_dati()
    
    valori_N_SIMBOLI = list(range(1, 11))  # Valori da 1 a 10
    valori_GIORNI_POS = list(range(5, 61, 5))  # Valori da 5 a 60 con step di 5
    valori_SL = list(np.arange(0.01, 0.1, 0.01))
    valori_TP = list(np.arange(0.1, 1, 0.05))

    space  = [
        Categorical(valori_N_SIMBOLI, name='N_SIMBOLI'),  
        Categorical(valori_GIORNI_POS, name='GIORNI_POS'),  
        Categorical(valori_SL, name='SL'),  
        Categorical(valori_TP, name='TP')
    ]

    @use_named_args(space)
    def objective(**params):
        return borsa.funzione_obiettivo(**params)

    risultato = gp_minimize(objective, space, n_calls=50, random_state=0)

    print(f"Migliori parametri: {risultato.x}")

    x_iters = risultato.x_iters  # Lista di parametri testati
    fun_values = risultato.func_vals  # Lista di valori della funzione obiettivo

    df_results = pd.DataFrame(x_iters, columns=['N_SIMBOLI', 'GIORNI_POS', 'SL', 'TP'])
    df_results['Valore_Funzione'] = fun_values
