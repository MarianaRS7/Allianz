import yfinance as yf
from datetime import datetime, timedelta
from googletrans import Translator
import numpy as np

# Inicializar el traductor
translator = Translator()

# Lista de nombres de ETFs y sus símbolos
etf_nombres = [
    "AZ QQQ NASDAQ 100",
    "AZ SPDR S&P 500 ETF TRUST",
    "AZ SPDR DJIA TRUST",
    "AZ VANGUARD EMERGING MARKET ETF",
    "AZ FINANCIAL SELECT SECTOR SPDR",
    "AZ HEALTH CARE SELECT SECTOR",
    "AZ DJ US HOME CONSTRUCT",
    "AZ SILVER TRUST",
    "AZ MSCI TAIWAN INDEX FD",
    "AZ MSCI UNITED KINGDOM",
    "AZ MSCI SOUTH KOREA IND",
    "AZ MSCI EMU",
    "AZ MSCI JAPAN INDEX FD",
    "AZ MSCI CANADA",
    "AZ MSCI GERMANY INDEX",
    "AZ MSCI AUSTRALIA INDEX",
    "AZ BARCLAYS AGGREGATE"
]

# Tickers correspondientes a los ETFs
etf_tickers = [
    "QQQ",
    "SPY",
    "DIA",
    "VWO",
    "XLF",
    "XLV",
    "ITB",
    "SLV",
    "EWT",
    "EWU",
    "EWY",
    "EZU",
    "EWJ",
    "EWC",
    "EWG",
    "EWA",
    "AGG"
]

def obtener_fechas_ultimos_diez_anos():
    """Obtiene las fechas de inicio y fin para los últimos 10 años."""
    fecha_fin = datetime.now()
    fecha_inicio = fecha_fin - timedelta(days=365 * 10)  # 10 años
    return fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d")

def descargar_datos_historicos(tickers):
    """Descarga los precios históricos de los últimos 10 años para una lista de tickers."""
    fecha_inicio, fecha_fin = obtener_fechas_ultimos_diez_anos()
    precios_historicos = {}
    
    for ticker in tickers:
        try:
            accion = yf.Ticker(ticker)
            datos = accion.history(start=fecha_inicio, end=fecha_fin)
            precios_historicos[ticker] = datos
        except Exception as e:
            print(f"Error al descargar datos para {ticker}: {e}")
            precios_historicos[ticker] = None
    
    return precios_historicos

def obtener_data(ticker):
    """Obtiene el nombre corto y la descripción larga de un ETF dado su ticker."""
    try:
        accion = yf.Ticker(ticker)
        info = accion.info
        nombre_corto = info.get('shortName', 'No disponible')
        descripcion_larga = info.get('longBusinessSummary', 'Descripción no disponible')
        descripcion_traducida = traducir_texto(descripcion_larga)  # Traducir la descripción
        return nombre_corto, descripcion_traducida
    except Exception as e:
        print(f"Error al obtener datos para {ticker}: {e}")
        return 'No disponible', 'Descripción no disponible'

def traducir_texto(texto):
    """Traduce un texto al español utilizando Google Translate."""
    try:
        traduccion = translator.translate(texto, dest='es')
        return traduccion.text
    except Exception as e:
        print(f"Error al traducir el texto: {e}")
        return 'Traducción no disponible'

def obtener_precio_actual(ticker):
    """Obtiene el precio de cierre más reciente de un ETF o acción dado su ticker."""
    try:
        # Crear el objeto del ticker
        accion = yf.Ticker(ticker)
        # Descargar el precio de cierre del último día de negociación
        precio_actual = accion.history(period='1d')['Close'].iloc[-1]
        return precio_actual
    except Exception as e:
        print(f"Error al obtener el precio actual para {ticker}: {e}")
        return None

def rendimiento_logaritmico(precios_historicos):
    """Calcula el rendimiento logarítmico anualizado a partir de los precios históricos."""
    precios = precios_historicos['Close']
    primer_precio = precios.iloc[0]
    ultimo_precio = precios.iloc[-1]
    
    rendimiento_log = np.log(ultimo_precio / primer_precio)
    rendimiento_log_anualizado = rendimiento_log / 10  # Dividir entre 10 años
    
    return rendimiento_log_anualizado

def calcular_riesgo_promedio(precios_historicos):
    """Calcula el riesgo promedio (desviación estándar anualizada) basado en precios de cierre históricos."""
    precios = precios_historicos['Close']
    rendimientos_diarios = np.log(precios / precios.shift(1)).dropna()
    desviacion_diaria = rendimientos_diarios.std()
    riesgo_promedio_anualizado = desviacion_diaria * np.sqrt(252)  # 252 días de negociación en un año
    return riesgo_promedio_anualizado

def calcular_ratio_riesgo_rendimiento(rendimiento_anualizado, riesgo_promedio):
    """Calcula el ratio riesgo-rendimiento."""
    if riesgo_promedio > 0:
        return rendimiento_anualizado / riesgo_promedio
    else:
        return None

def rendimiento_y_riesgo_por_periodo(precios_historicos, periodo):
    """Calcula el rendimiento y riesgo para un periodo específico."""
    try:
        if periodo == '1m':
            datos_periodo = precios_historicos.last('1M')
        elif periodo == '3m':
            datos_periodo = precios_historicos.last('3M')
        elif periodo == '6m':
            datos_periodo = precios_historicos.last('6M')
        elif periodo == '1y':
            datos_periodo = precios_historicos.last('1Y')
        elif periodo == 'YTD':
            datos_periodo = precios_historicos[precios_historicos.index >= datetime.now().replace(month=1, day=1)]
        elif periodo == '3y':
            datos_periodo = precios_historicos.last('3Y')
        elif periodo == '5y':
            datos_periodo = precios_historicos.last('5Y')
        elif periodo == '10y':
            datos_periodo = precios_historicos.last('10Y')
        else:
            raise ValueError("Periodo no reconocido.")

        # Calcular el rendimiento logarítmico
        rendimiento_log = np.log(datos_periodo['Close'].iloc[-1] / datos_periodo['Close'].iloc[0])
        rendimiento_anualizado = rendimiento_log / (datos_periodo.shape[0] / 252)  # Ajustar por días de negociación

        # Calcular el riesgo
        rendimientos_diarios = np.log(datos_periodo['Close'] / datos_periodo['Close'].shift(1)).dropna()
        desviacion_diaria = rendimientos_diarios.std()
        riesgo_anualizado = desviacion_diaria * np.sqrt(252)

        return rendimiento_anualizado, riesgo_anualizado
    except Exception as e:
        print(f"Error al calcular rendimiento y riesgo para el periodo {periodo}: {e}")
        return None, None

# Variable para almacenar la información de los ETFs
ETFs_Data = []

# Descargar precios históricos para todos los tickers
precios_historicos_todos = descargar_datos_historicos(etf_tickers)

# Iterar sobre los ETFs y obtener la información
for nombre, ticker in zip(etf_nombres, etf_tickers):
    nombre_corto, descripcion_larga = obtener_data(ticker)
    
    # Obtener los precios históricos del ticker actual
    precios_historicos = precios_historicos_todos.get(ticker)
    
    # Obtener el precio actual
    precio_actual = obtener_precio_actual(ticker)

    # Calcular el rendimiento logarítmico anualizado
    if precios_historicos is not None and not precios_historicos.empty:
        rendimiento_log_geom = rendimiento_logaritmico(precios_historicos)
        riesgo_promedio = calcular_riesgo_promedio(precios_historicos)
        ratio_riesgo_rendimiento = calcular_ratio_riesgo_rendimiento(rendimiento_log_geom, riesgo_promedio)

        # Calcular rendimiento y riesgo para diferentes periodos
        periodos = ['1m', '3m', '6m', '1y', 'YTD', '3y', '5y', '10y']
        rendimientos = {}
        riesgos = {}
        
        for periodo in periodos:
            rendimiento, riesgo = rendimiento_y_riesgo_por_periodo(precios_historicos, periodo)
            rendimientos[periodo] = rendimiento
            riesgos[periodo] = riesgo

    else:
        rendimiento_log_geom = None
        riesgo_promedio = None
        ratio_riesgo_rendimiento = None
        rendimientos = {periodo: None for periodo in periodos}
        riesgos = {periodo: None for periodo in periodos}
    
    # Añadir la información a la lista de ETFs
    ETFs_Data.append({
        "nombre": nombre,
        "simbolo": ticker,
        "nombre_corto": nombre_corto,
        "descripcion_larga": descripcion_larga,
        "precios_historicos": precios_historicos,
        "precio_actual": precio_actual,
        "rendimiento_log_geom": rendimiento_log_geom,
        "riesgo_promedio": riesgo_promedio,
        "ratio_riesgo_rendimiento": ratio_riesgo_rendimiento,
        "rendimientos": rendimientos,
        "riesgos": riesgos
    })
