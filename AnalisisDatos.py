#!/usr/bin/env python
# coding: utf-8

# # 1. Carga de librerías y datos

# In[136]:


# Carga de librerias
import pandas as pd
import math
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa as tsm
from pmdarima.arima import auto_arima
import datetime


# In[5]:


# Descarga de datos
datosaapl = pd.read_excel("https://datos-aapl.s3.amazonaws.com/datosAAPL.xlsx")
datosaapl.head()


# In[42]:


# Función para calcular ACF y PACF
def acf1(x, nlags=None, acf_type="correlation", pacf=False, ax=None, **kwargs):
    lags = np.arange(1, nlags + 1)

    if pacf:
        if acf_type == "correlation":
            values = sm.tsa.pacf(x, nlags=nlags)[1:]
            ylabel = "PACF"
    else:
        if acf_type == "correlation":
            values = sm.tsa.acf(x, nlags=nlags, fft=False)[1:]
            ylabel = "ACF"
        elif acf_type == "covariance":
            values = sm.tsa.acovf(x, nlag=nlags)[1:]
            ylabel = "ACoV"

    if ax is None:
        ax = plt.gca()
 
    ax.bar(lags, values, **kwargs)
    ax.axhline(0, color="black", linewidth=1)
    if acf_type == "correlation":
        conf_level = 1.96 / np.sqrt(x.shape[0])
        ax.axhline(conf_level, color="red", linestyle="--", linewidth=1)
        ax.axhline(-conf_level, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("LAG")
    ax.set_ylabel(ylabel)

    return ax


# # 2. Ajuste del formato de la serie de tiempo

# In[6]:


# Tipo de columnas actuales
datosaapl.info()


# In[7]:


# Poniendo de índice del DataFrame a la fecha
datosaapl.set_index('Date', inplace = True)
datosaapl.info()


# In[8]:


datosaapl.head() # El DataFrame ya tiene un DatetimeIndex


# In[162]:


datosaapl_custom_business_day = datosaapl.asfreq('C') #https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
datosaapl_custom_business_day[datosaapl_custom_business_day['Adj Close'].isnull()]


# # 3. Visualización

# In[36]:


# Visualización de la serie de tiempo
fig = px.line(x = datosaapl.index, y = datosaapl["Adj Close"], 
              title = "Apple Share price", 
              labels = {"x": "Date", "y": "Price"}, template = "plotly_dark")
fig.show()


# In[20]:


# Prueba de estacionariedad (alfa = 0.05)

# https://www.statology.org/dickey-fuller-test-python/

# Ho: La serie de tiempo no es estacionaria
# Ha: La serie de tiempo es estacionaria

adfuller(datosaapl)[1] # p-value > 0.05, la serie aún no es estacionaria


# # 4. Transformación de la serie

# In[40]:


# Transformaciones de la serie

# Diferenciación para eliminar tendencia
# Transformación logarítmica para reducir variabilidad en los valores extremos
diff_log_appl = datosaapl.apply(np.log).diff().dropna()

# Visualización de la serie de tiempo diferenciada
fig = px.line(x = diff_log_appl.index, y = diff_log_appl["Adj Close"], 
              title = "Apple Share price (transformed)", 
              labels = {"x": "Date", "y": "Price"}, template = "seaborn")
fig.show()


# In[55]:


# Prueba de estacionariedad a la serie diferenciada
adfuller(diff_log_appl)[1] # p-value < 0.05, la serie ya puede ser considerada como estacionaria


# # 5. ACF y PACF de la serie transformada

# In[59]:


# ACF
fig, axes = plt.subplots(nrows = 2, figsize=(12, 8))

acf1(diff_log_appl, nlags = 48, ax = axes[0], width = .4)
axes[0].set_title("Apple share price (transformed) - ACF")

# PACF
acf1(diff_log_appl, nlags = 48, ax = axes[1], width = .4, pacf = True)
axes[1].set_title("Apple share price (transformed) - PACF")

fig.tight_layout()
plt.show()


# # 6. Modelos propuestos

# No se observa patrón estacional ni en la ACF (que indicaría estacionalidad en el término MA) ni en la PACF (que indicaría estacionalidad en el término AR).
# #### Lags con mayor significancia por encima del umbral:
# * En la ACF: 9, 8, 7 y 2 (Niveles tentativos del parámetro "q" de MA)
# * En la PACF: 9, 8, 7 y 2 (Niveles tentativos del parámetro "p" de AR)
# 
# Se asume que los retrasos de mayor orden son no significativos para el armado del modelo. Se utilizará como criterio el modelo que proporcione un menor AIC, aunque también podría utilizar procedimientos más a detalle como backtesting.

# In[62]:


# Serie logarítmica


# In[64]:


aapl_log = datosaapl.apply(np.log)
aapl_log.head()


# Se iniciará probando un modelo ARIMA con rezagos específicos (2, 7, 8 y 9), tanto para los niveles de AR como MA. Como la serie fue diferenciada una única vez, el valor de I = 1.

# In[65]:


p_orders = (2, 7, 8, 9)
q_orders = (2, 7, 8, 9)


# In[107]:


# Modelo 1 (rezagos específicos)
# https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

modelo_uno = sm.tsa.SARIMAX(aapl_log, order = (p_orders, 1, q_orders), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_uno.summary()


# A continuación, se probará con modelos en dónde p = q, para los órdenes (9, 8, 7, 2). Se optará por no probar modelos en donde p != q por 2 motivos: el primero es porque el problema se volvería extenso (demasiados modelos por probar), mientras que el segundo es que según las gráficas de ACF y PACF, cuando un rezago es significativo en el ACF (que da indicios del nivel q de MA), también es significativo en el PACF (que da indicios del nivel p de AR), para los rezagos de temporalidad menores al 10° rezago.

# In[99]:


# Modelo 2 (rezagos altos)
modelo_dos = sm.tsa.SARIMAX(aapl_log, order = (9, 1, 9), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_dos.summary()


# In[108]:


# Modelo 3
modelo_tres = sm.tsa.SARIMAX(aapl_log, order = (8, 1, 8), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_tres.summary()


# In[109]:


# Modelo 4
modelo_cuatro = sm.tsa.SARIMAX(aapl_log, order = (7, 1, 7), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_cuatro.summary()


# In[110]:


# Modelo 5
modelo_cinco = sm.tsa.SARIMAX(aapl_log, order = (2, 1, 2), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_cinco.summary()


# Finalmente, una alternativa es utilizar la función auto_arima del paquete pmdarima para determinar los niveles de p y q.

# In[112]:


auto_arima(aapl_log, seasonal = False)


# In[113]:


# Probando con los niveles indicados por la función auto_arima
# Modelo 6
modelo_seis = sm.tsa.SARIMAX(aapl_log, order = (1, 1, 0), trend = None).fit()
# El warning de date index se debe a que se está trabajando con trading days
modelo_seis.summary()


# Se observa que el modelo que presenta el mejor AIC es el modelo uno con rezagos específicos. Sin embargo, la prueba de Ljung-box a los residuales exhibe una falta de ajuste del modelo al momento de examinar las autocorrelaciones de los residuales (p-value = 0 < alfa = 0.05).
# 
# Por ende, se opta por usar el segundo mejor modelo, el modelo dos --> ARIMA(9,1,9), que brinda el segundo menor AIC posible y no exhibe una falta de ajuste (p-value Ljung-box test = 0.84 > alfa = 0.05).

# # 7. Diagnóstico del modelo

# In[116]:


fig = modelo_dos.plot_diagnostics(figsize=(10, 7))
fig.tight_layout()
plt.show()


# * Los residuales tienen media constante y cercana a 0, y aparentan tener media constante
# * El Q-Q plot evidencia un ajuste regular al modelo, considerando que esta gráfica tiende a aumentar la diferencia en los extremos
# * Además, el correlograma no evidencia autocorrelación entre los residuales
# 

# # 8. Pronóstico

# In[236]:


# Dado que los datos son mensuales, se ejemplifica el pronóstico de una semana en el nivel original

forecast = modelo_dos.get_forecast(7)
forecast_index = pd.date_range(datosaapl.index[-1] + datetime.timedelta(days=1),periods=7, freq='d')

forecast_mean = np.exp(modelo_dos.get_forecast(7) .predicted_mean)
forecast_mean.index = forecast_index

se_mean = np.exp(modelo_dos.get_forecast(7).se_mean)
se_mean.index = forecast_index


# In[239]:


# Dado que los datos son diarios, se ejemplifica el pronóstico de una semana
#https://www.statsmodels.org/v0.13.0/examples/notebooks/generated/statespace_forecasting.html
#https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast.html#statsmodels.tsa.statespace.sarimax.SARIMAXResults.get_forecast
plt.plot(datosaapl.index[-10:], datosaapl["Adj Close"][-10:])
forecast_mean.plot(marker="o")
plt.plot(forecast_mean + se_mean, color="red", linestyle="--")
plt.plot(forecast_mean - se_mean, color="red", linestyle="--")
plt.show()                     

