#!/bin/bash
git checkout -- .
git pull origin main
rm -r bayesian_search/
rm -r dati/
pip install keras_tuner
pip install tensorflow
pip install yfinance
pip install pandas_ta
pip install plotly
pip install scikit-learn
pip install pyarrow
pip install fastparquet
pip install imbalanced-learn
pip install numpy --upgrade