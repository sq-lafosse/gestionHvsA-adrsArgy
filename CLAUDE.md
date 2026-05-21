# CLAUDE.md — gestionHvsA-adrsArgy

## Qué es este proyecto

Sistema de **Monthly Tactical Asset Allocation** sobre un universo de activos argentinos denominados en USD.

El objetivo académico es comparar el desempeño de gestión **humana vs. algorítmica** en dos períodos de mercado opuestos: uno alcista y uno bajista.

El modelo **no sabe de antemano** cuál período es alcista o bajista. Toma decisiones mes a mes, en base a lo que ve: precios históricos hasta ese mes, datos macro disponibles y noticias curadas de ese mes. Es un observador que va decidiendo, no un oráculo.

El informe final está dirigido a lectores no técnicos, pero con rigor académico.

---

## Universo de activos

### Renta variable — ADRs argentinos
| Ticker | Empresa |
|--------|---------|
| YPF | YPF S.A. |
| GGAL | Grupo Financiero Galicia |
| BMA | Banco Macro |
| PAM | Pampa Energía |
| TGS | Transportadora de Gas del Sur |
| CEPU | Central Puerto |
| EDN | Empresa Distribuidora Norte |
| LOMA | Loma Negra |
| CRESY | Cresud |
| IRS | IRSA Inversiones y Representaciones |

### Renta fija soberana
| Ticker | Instrumento |
|--------|-------------|
| AL30D | Bono soberano argentino en USD — scraping automático (ver módulo data) |

### Reglas sobre el universo
- El universo define los **candidatos posibles**, no los activos obligatorios.
- Cada mes el modelo puede asignar **0% a cualquier ADR** si las señales no lo justifican.
- Un activo descartado un mes **puede reincorporarse** el mes siguiente si las señales cambian.
- Toda exclusión o reincorporación queda registrada en el log con la señal que la motivó.
- El universo de candidatos es inmutable: no se agregan ni se quitan tickers durante el proyecto.

---

## Lógica temporal — Walk-Forward puro

```
2015-01  →  2023-12   PERÍODO IN-SAMPLE
                       Cache histórico. Se descarga UNA SOLA VEZ y queda inmutable.
                       Aquí se entrena PCA + SVM.
                       Si un activo no tiene datos desde 2015, se usa desde donde esté disponible.
                       Fechas dispares entre activos son aceptables — se documenta en el log.

2024-01  →  adelante   PERÍODO LIVE (walk-forward mes a mes)
                       El sistema agrega UN MES a la vez con información nueva.
                       Nunca re-descarga el cache histórico ya almacenado.
```

**Regla crítica:** al correr el período live, el modelo ve **solo** lo que existiría en esa fecha real:
- Precios hasta el cierre del mes en curso
- Datos macro disponibles hasta ese mes
- Noticias curadas de ese mes (PDFs/TXT en `data/news/`)
- Sin acceso a meses futuros — zero data leakage

El código **nunca recibe como parámetro** si un período es "bull" o "bear". Esa es una etiqueta que se asigna **a posteriori** para el análisis académico.

---

## Dos modos de ejecución

### Modo histórico (corre UNA sola vez)
- Descarga y almacena toda la serie disponible hasta diciembre 2023
- Llena `data/cache/` con precios, AL30D, CCL y datos macro
- Una vez generado, este cache es **inmutable**
- Si un activo tiene datos desde 2018 y otro desde 2015: se acepta, se documenta, se continúa

### Modo live (corre mes a mes desde enero 2024)
- Lee el cache histórico existente
- Agrega solo el mes nuevo al final
- Toma decisiones de allocación con la información disponible hasta ese mes
- Guarda resultados en `results/{año}/`

---

## Fuentes de datos — todo por scraping automático

**No se usan CSVs manuales en ningún momento.**

| Dato | Fuente principal | Fallback | Almacenamiento |
|------|-----------------|----------|----------------|
| Precios ADRs | yfinance (`Adj Close`, semanal) | scraping alternativo | `data/cache/prices/` |
| AL30D en USD | Scraping automático (Investing.com u otras) | fuente alternativa disponible | `data/cache/prices/` |
| CCL histórico | Scraping automático (Ámbito, Investing u otras) | fuente alternativa disponible | `data/cache/macro/` |
| Merval (`^MERV`) | yfinance | scraping alternativo | `data/cache/prices/` |
| Datos macro AR | Scraping automático (ver detalle abajo) | fuente alternativa disponible | `data/cache/macro/` |

### Lógica de fallback para scraping
Si la fuente principal falla → intenta fuente alternativa → si ambas fallan → usa último dato cacheado disponible y registra warning en log. El pipeline nunca se detiene por falla de una fuente.

---

## Datos macroeconómicos de Argentina

Se descargan y almacenan en `data/cache/macro/` como contexto para el clasificador de régimen.
Período: post-pandemia (desde 2020 aprox.) hasta el mes en curso.

| Variable | Fuente sugerida |
|----------|----------------|
| Inflación mensual (IPC) | INDEC / scraping |
| Tipo de cambio CCL histórico | Ámbito / Investing |
| Riesgo país (EMBI Argentina) | Ámbito / scraping |
| Reservas BCRA | BCRA / scraping |
| Tipo de cambio oficial | BCRA / scraping |

Estos datos enriquecen el contexto del clasificador de régimen y el análisis académico del paper.

---

## Benchmark

| Benchmark | Ticker / Fuente | Conversión |
|-----------|----------------|------------|
| EW Buy & Hold ADRs | Los 10 ADRs en equal weight estático | Ya en USD |
| AL30D estático | Scraping automático | Ya en USD |
| Merval en USD | `^MERV` via yfinance | Dividido por CCL histórico scrapeado |

**Tipo de cambio para Merval:** se usa **CCL (Contado con Liquidación)** — es el tipo de cambio relevante para inversores que operan ADRs y activos argentinos en USD. Es el estándar en research de mercados emergentes argentinos.

---

## Lógica de asignación

### Régimen Risk-On
- Capital en **equal weight** entre los ADRs con señal positiva ese mes
- ADRs con señal negativa → 0% ese mes (pueden volver el siguiente)
- AL30D: 0%

### Régimen Risk-Off
- **80%** en ADRs activos (equal weight entre los con señal positiva)
- **20%** AL30D

### Rebalanceo
- Se dispara **solo cuando cambia el régimen** (Risk-On ↔ Risk-Off)
- Pesos siempre suman 100%
- Sin cash, sin short selling, sin apalancamiento

---

## Frecuencia operativa

| Elemento | Frecuencia |
|----------|------------|
| Datos de entrada | Semanal |
| Decisiones de allocación | Mensual |
| Rebalanceo | Solo al cambio de régimen |

---

## Señales técnicas

Calculadas sobre el **retorno agregado del portfolio** (promedio del universo activo ese mes):

- SMA 20
- SMA 30
- Momentum
- Volatilidad realizada

Las señales individuales por activo determinan si ese ADR entra o no ese mes.

---

## Clasificación de régimen

- **PCA** para reducción de dimensionalidad de las señales técnicas y macro
- **SVM** para clasificación binaria: Risk-On / Risk-Off
- Entrenado en período in-sample: 2015-01 → 2023-12
- Aplicado walk-forward mes a mes desde enero 2024

---

## NLP — Sentimiento

- Score mensual agregado (positivo / negativo) sobre el universo completo
- Noticias curadas **manualmente**: PDFs y TXT en `data/news/{año}/{mes}/`
- Modelo: **FinBERT**
- El score se alimenta como input adicional al clasificador de régimen
- Integración exacta con SVM: a definir al construir el módulo

---

## Stack tecnológico

```
Python 3.11+
pandas / numpy          → manipulación de datos
scikit-learn            → PCA, SVM
vectorbt                → motor de backtest
transformers / FinBERT  → NLP
matplotlib              → visualización estilo académico (PNG/PDF para el paper)
yfinance                → descarga de precios ADRs y Merval
requests / BeautifulSoup → scraping de AL30D, CCL y datos macro
python-dotenv           → variables de entorno
pyyaml                  → configuración
```

### Visualización
- **matplotlib** con estilo académico
- Outputs: PNG/PDF exportables para incluir directamente en el informe
- Tablas de métricas: impresas en terminal y exportadas como CSV
- Sin dashboards interactivos, sin HTML

---

## Configuración del proyecto

```
config/
├── assets.yaml      → lista de tickers y metadatos
├── periods.yaml     → fechas in-sample y out-of-sample
└── settings.yaml    → parámetros del modelo (ventanas SMA, umbrales, etc.)
```

Variables sensibles → `.env`, nunca en YAML ni en el código.

---

## Estructura del proyecto

```
gestionHvsA-adrsArgy/
│
├── config/
│   ├── assets.yaml
│   ├── periods.yaml
│   └── settings.yaml
│
├── data/
│   ├── cache/
│   │   ├── prices/          ← ADRs, AL30D, Merval (cache histórico inmutable post-primera-corrida)
│   │   └── macro/           ← CCL, inflación, riesgo país, reservas BCRA
│   ├── news/
│   │   ├── 2024/
│   │   │   ├── 01/          ← noticias curadas enero 2024
│   │   │   ├── 02/
│   │   │   └── ...
│   │   └── 2025/
│   │       ├── 01/
│   │       └── ...
│   └── exports/             ← datos procesados listos para análisis
│
├── notebooks/               ← exploración y validación
│
├── results/
│   ├── 2024/                ← outputs período out-of-sample bull
│   └── 2025/                ← outputs período out-of-sample bear
│
├── src/
│   ├── data/                ← descarga, scraping, validación y cache
│   ├── features/            ← cálculo de señales técnicas
│   ├── signals/             ← generación de señales y clasificación de régimen
│   ├── allocation/          ← motor de asignación de pesos
│   ├── backtest/            ← walk-forward con vectorbt
│   ├── metrics/             ← Sharpe, drawdown, Treynor, Jensen
│   ├── nlp/                 ← pipeline FinBERT y score mensual
│   └── utils/               ← logging, helpers, I/O
│
├── tests/
│
├── main.py                  ← pipeline integrado, outputs separados por año
├── requirements.txt
├── .env                     ← API keys (NO commitear)
├── .gitignore
└── CLAUDE.md
```

---

## Pipeline — main.py

`main.py` contiene el pipeline completo integrado. Flujo secuencial:

```
[MODO HISTÓRICO — solo primera corrida]
1. Scraping y descarga de serie histórica 2015→2023 (precios + macro + CCL)
2. Almacenamiento en data/cache/ (inmutable después de esta corrida)
3. Entrenamiento PCA + SVM sobre período in-sample

[MODO LIVE — corre mes a mes desde enero 2024]
4. Lee cache histórico + agrega datos del mes en curso
5. Descarga noticias curadas del mes (data/news/)
6. Cálculo de señales técnicas sobre datos disponibles hasta ese mes
7. Procesamiento NLP del mes (FinBERT → score de sentimiento)
8. Clasificación de régimen (PCA + SVM)
9. Decisión de allocación mensual (pesos por activo)
10. Registro en log (activos incluidos, pesos, régimen, señales)
11. Simulación walk-forward acumulada (vectorbt)
12. Cálculo de métricas de performance vs. benchmarks
13. Exportación en results/{año}/ + figuras matplotlib
```

**Regla absoluta:** ningún paso puede usar información posterior a la fecha del mes en curso.

---

## Outputs y persistencia

Por cada corrida en `results/{año}/`:
- `portfolio_weights.csv` — pesos mensuales por activo
- `performance_metrics.csv` — métricas vs. benchmarks
- `snapshot_{timestamp}.json` — snapshot reproducible (parámetros + datos usados)
- `figures/` — gráficos PNG/PDF para el informe
- `logs/run_{timestamp}.log` — trazabilidad completa de decisiones

---

## Principios de desarrollo (NO negociables)

1. **Sin data leakage** — el modelo solo ve lo que existiría en esa fecha real
2. **El modelo no conoce los labels bull/bear** — son etiquetas analíticas a posteriori
3. **Sin CSVs manuales** — todo dato llega por scraping o API automática
4. **Cache histórico inmutable** — una vez generado, no se re-descarga
5. **Reproducibilidad** — toda corrida genera snapshot JSON con parámetros y datos usados
6. **Trazabilidad** — toda decisión de portfolio queda en el log con su señal
7. **Modularidad** — cada módulo de `src/` tiene una sola responsabilidad
8. **Un solo `src/`** — separación de períodos en `config/` y `results/`, nunca en el código
9. **Universo inmutable** — los candidatos no cambian; solo cambian los pesos mensuales
10. **Fechas dispares aceptadas** — si un activo tiene menos historia, se usa lo disponible y se documenta

---

## Contexto académico

- El paper compara gestión **humana** (sesgos conductuales) vs. **algorítmica** (sistemática)
- Los períodos bull/bear son etiquetas **analíticas a posteriori**, no inputs del modelo
- El informe está dirigido a lectores **no técnicos** con profundidad académica
- Toda decisión de diseño debe poder **justificarse metodológicamente** en el paper

---

## Orden de construcción sugerido

```
1. src/data/          → scraping, descarga, cache (histórico + live)
2. src/features/      → señales técnicas
3. src/signals/       → clasificación de régimen (PCA + SVM)
4. src/allocation/    → motor de pesos
5. src/backtest/      → walk-forward con vectorbt
6. src/metrics/       → métricas y benchmarks (incluye conversión Merval a USD con CCL)
7. src/nlp/           → FinBERT y score de sentimiento
8. src/utils/         → logging y helpers (se construye en paralelo desde el inicio)
9. main.py            → integración final de los dos modos
```