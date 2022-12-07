# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 18:55:20 2022

@author: Luis Jerí
"""

# =============================================================================
# Actualizar dash antes de correr el archivo
# pip install dash --upgrade (en la consola)
# print(dash.__version__)
# =============================================================================

# =============================================================================
# Bibliografia consultada
# =============================================================================

# https://plotly.com/python/time-series/
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.base.prediction.PredictionResults.html
# https://community.plotly.com/t/how-to-plot-multiple-lines-on-the-same-y-axis-using-plotly-express/29219/2
# https://stackoverflow.com/questions/61218501/plotly-how-to-show-legend-in-single-trace-scatterplot-with-plotly-express
# https://github.com/plotly/plotly.py/issues/2457
# https://stackoverflow.com/questions/30280856/populating-a-dictionary-using-for-loops-python

# =============================================================================
# Público objetivo: usuarios que deseen conocer una primera aproximación del 
# pronóstico del precio de la acción de Apple en la bolsa NASDAQ. Esto tomando
# en cuenta que un modelo ARIMA no incluye aspecto típicamente implicados en el
# pronóstico de series financieras (efecto ARCH-GARCH, medidas de riesgo y 
# volatilidad, modelos multivariados tipo VAR, entre otros). Sin embargo, el 
# modelo ARIMA propuesto representa una aproximación inicial al pronóstico, que 
# es sencilla de comprender.
# =============================================================================

# =============================================================================
# Importación de librerías
# =============================================================================
from dash import Dash, dcc, html, Input, Output
import dash

import pandas as pd # Tratamiento de datos
import numpy as np
import plotly.express as px # Visualización de datos

import statsmodels.api as sm # Modelo ARIMA
import datetime # Generación de índice de fechas

# =============================================================================
# Descarga de datos
# =============================================================================

datosaapl = pd.read_excel("https://datos-aapl.s3.amazonaws.com/datosAAPL.xlsx")
datosaapl.set_index('Date', inplace = True) # Colocando el DateTimeIndex
aapl_log = datosaapl.apply(np.log) # Transformación logarítmica

# =============================================================================
# Ajuste del modelo final
# =============================================================================
modelo_final = sm.tsa.SARIMAX(aapl_log, order = (9, 1, 9), trend = None).fit()

# =============================================================================
# Diccionarios para el tablero
# =============================================================================

# Primera fecha del pronóstico hacia adelante
first_date_forecasted = datosaapl.index[-1] + datetime.timedelta(days=1)

# Diccionario para datos históricos 
historical_dicts = {}
keys = range(datosaapl.index.size)
for i in keys:
    historical_dicts[datosaapl.index[i].date().strftime("%d/%m/%Y")] = datosaapl.index[i]
    
# Diccionario para fecha final del forecast (se asume máximo 1 año de pronóstico) (para rellenar el dropdown)
maximum_date_forecasted = datosaapl.index[-1] + datetime.timedelta(days = 365) # 1 año
maximum_period = pd.date_range(start = first_date_forecasted, end = maximum_date_forecasted, freq = "d")

forecast_dicts = {}
keys_forecast = range(maximum_period.size)
for i in keys_forecast:
    forecast_dicts[maximum_period[i].date().strftime("%d/%m/%Y")] = maximum_period[i]

# =============================================================================
# Creación del objeto app en Dash
# =============================================================================

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#app.run_server(debug=True) # Debugging
server = app.server

# =============================================================================
# Creación de los elementos del layout del tablero en Dash
# =============================================================================
app.layout = html.Div([
html.Div([
    # Cabecera del tablero y descripción inicial
    html.H2(children = "Pronóstico del precio de la acción de Apple", dir = "ltr",
            style={'color': '#2e3033', "background-color": "#f3f3f3", "font-weight": "bold", 'text-align':'center'}),
    html.H4("Pronóstico del adjusted price per day usando un modelo ARIMA(9,1,9)", 
            style={'color': '#f3f3f3', "background-color": "#2e3033", "font-weight": "bold", 'text-align':'center'}),
    html.Hr()
]),
html.Div([
    #Dropdown de fecha inicial de visualización
    dcc.Dropdown(
        options = historical_dicts,
        value = datosaapl.index[0],
    id = "historical_dropdown"
    ),
    html.Div("Inicio visualización", style={'text-align':'center'})
    ],
        style={'width': '50%', 'display': 'inline-block'}),
html.Div([
    #Dropdown de fecha final del forecast
    dcc.Dropdown(
        options = forecast_dicts,
        value = maximum_period[0], # Inicialización: escogiendo días de pronóstico en vez de fecha fin de pronóstico
    id = "forecast_dropdown"
    ),
    html.Div("Fin pronóstico", style={'text-align':'center'})
    ],
        style={'width': '50%', 'display': 'inline-block'}),
# Gráfico
html.Hr(),
dcc.Graph(id='grafico_forecast'),

# Mensajes (instrucciones y recomendaciones de uso)
html.Div([
    html.Hr(),
    html.H2("Esta es una primera aproximación al precio futuro de la acción", 
                 style={'color': '#2e3033', "font-weight": "bold"}),  
    html.H3("En caso de requerir un pronóstico más certero, recurrir a métodos econométricos o de riesgos", 
                 style={'color': '#2e3033', "font-weight": "bold"}),        
    html.Hr()
    ])
])

# =============================================================================
# Definición de la función junto al decorador @app.callback
# =============================================================================           

@app.callback(
    Output('grafico_forecast', 'figure'),
    [Input('historical_dropdown', 'value'),
     Input('forecast_dropdown', 'value')])
def update_figura(historical_dropdown, forecast_dropdown):
    # =============================================================================
    # Generación del pronóstico hacia adelante
    # =============================================================================
    
    first_date_forecasted = datosaapl.index[-1] + datetime.timedelta(days=1)
    # Última fecha del pronóstico hacia adelante
    last_date_forecasted = forecast_dropdown
 
    # Periodo de pronóstico hacia adelante
    forecast_period = pd.date_range(start = first_date_forecasted, end = last_date_forecasted, freq = "d")
    # Generación del pronóstico hacia adelante
    forecast_appl = modelo_final.get_forecast(forecast_period.size)

    # Valor esperado del pronóstico
    forecast_mean_appl = np.exp(forecast_appl.predicted_mean)
    forecast_mean_appl.index = forecast_period

    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.base.prediction.PredictionResults.html
    # Desviación estándar del pronóstico
    se_mean_appl = np.exp(forecast_appl.se_mean)
    se_mean_appl.index = forecast_period
    
    # =============================================================================
    # Generación de visualizaciones
    # =============================================================================

    # Fecha desde la cual el usuario desea visualizar los datos históricos:
    start_date = historical_dropdown
    datosaapl_viz = datosaapl[datosaapl.index > start_date] # Filtrado para visualización

    # Historical graph
    fig = px.line(x = datosaapl_viz.index, y = datosaapl_viz["Adj Close"], 
                  title = "Apple Share price", 
                  labels = {"x": "Date", "y": "Price"}, template = "plotly_dark")
    fig['data'][0]['showlegend']=True
    fig['data'][0]['name']='Historical'
    # Forecast graph
    fig.add_scatter(x = forecast_mean_appl.index, y = forecast_mean_appl + se_mean_appl, name = "Forecast mean + SE")
    fig.add_scatter(x = forecast_mean_appl.index, y = forecast_mean_appl, name = "Forecast mean")
    fig.add_scatter(x = forecast_mean_appl.index, y = forecast_mean_appl - se_mean_appl, name = "Forecast mean - SE")
    
    return fig 
    
# =============================================================================
# Ejecución del archivo            
# =============================================================================

if __name__ == '__main__':
    app.run_server(debug=True)

