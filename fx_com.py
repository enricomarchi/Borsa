import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.subplots as sp
import yfinance as yf

def grafico(risultato, err):
    target = go.Scatter(
        x = risultato.index,
        y = risultato['Target'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 0, .9)'),
        name = 'Target'
    )
    
    previsione = go.Scatter(
        x = risultato.index,
        y = risultato['Previsione'],
        mode = 'lines',
        line = dict(color='rgba(0, 0, 250, .9)'),
        name = 'Previsione'
    )
    err_meno = go.Scatter(
        x = risultato.index,
        y = risultato['Previsione'] - err, 
        mode = 'lines',
        line = dict(
            color='rgba(0, 0, 250, .9)',
            width=1
        ),
        connectgaps = False,
        name = 'err-'
    )                
    err_piu = go.Scatter(
        x = risultato.index,
        y = risultato['Previsione'] + err, 
        mode = 'lines',
        line = dict(
            color='rgba(0, 0, 250, .9)',
            width=1
        ),
        fill='tonexty',
        fillcolor='rgba(0, 0, 250, .2)', # Puoi cambiare il valore di alpha per regolare la trasparenza
        connectgaps = False,
        name = 'err+'
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
    fig.add_trace(target, row=1, col=1)
    fig.add_trace(previsione, row=1, col=1)
    fig.add_trace(err_meno, row=1, col=1)
    fig.add_trace(err_piu, row=1, col=1)
    pyo.plot(fig, filename="regressione.html", auto_open=True)

