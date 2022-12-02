# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf

aapl = yf.download('AAPL', start='2017-11-28', end='2022-11-25', progress=False, keepna=False) # Descarga de datos
precio = list(aapl.columns)[4] # Adjusted prices
aapl = pd.DataFrame(aapl[precio])
writer = pd.ExcelWriter('datosAAPL.xlsx', engine='xlsxwriter')
aapl.to_excel(writer, sheet_name = 'APPLE', index = True)
writer.save()

spy = yf.download('SPY', start='2017-11-28', end='2022-11-25', progress=False, keepna=False) # Descarga de datos
precio = list(spy.columns)[4] # Adjusted prices
spy = pd.DataFrame(spy[precio])
writer = pd.ExcelWriter('datosSPY.xlsx', engine='xlsxwriter')
spy.to_excel(writer, sheet_name = 'SPY', index = True)
writer.save()

