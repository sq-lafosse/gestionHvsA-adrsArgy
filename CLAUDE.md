# CLAUDE.md — gestionHvsA-adrsArgy

## Qué es este proyecto

Sistema de **Monthly Tactical Asset Allocation** sobre un universo de activos argentinos denominados en USD.

El objetivo académico es comparar el desempeño de gestión **humana vs. algorítmica** en dos períodos de mercado opuestos: uno alcista y uno bajista.

El modelo **no sabe de antemano** cuál período es alcista o bajista. Toma decisiones mes a mes, en base a lo que ve: precios históricos hasta ese mes, datos macro disponibles y noticias curadas de ese mes. Es un observador que va decidiendo, no un oráculo.

El informe final está dirigido a lectores no técnicos, pero con rigor académico.

---

## Universo de activos

### Renta variable — ADRs argentinos (100% equity)
| Ticker | Empresa | Sector |
|--------|---------|--------|
| YPF | YPF S.A. | Energía petrolera |
| GGAL | Grupo Financiero Galicia | Bancario |
| BMA | Banco Macro | Bancario |
| PAM | Pampa Energía | Energía eléctrica |
| TGS | Transportadora de Gas del Sur | Infraestructura energética |
| CEPU | Central Puerto | Generación eléctrica |
| EDN | Empresa Distribuidora Norte | Distribución eléctrica |
| LOMA | Loma Negra | Construcción / materiales |
| CRESY | Cresud | Agro / Real Estate |
| IRS | IRSA Inversiones y Representaciones | Real Estate |

**El portfolio es 100% equity. No hay instrumentos de renta fija.**
AL30D fue eliminado del universo por falta de datos históricos en fuentes reguladas equivalentes a SEC.

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

2024-01  →  2024-12   PERÍODO OUT-OF-SAMPLE — Bull market
2025-01  →  2025-12   PERÍODO OUT-OF-SAMPLE — Bear market
```

**Regla crítica:** al correr el período live, el modelo ve **solo** lo que existiría en esa fecha real:
- Precios hasta el cierre del mes en curso
- Datos macro disponibles hasta ese mes
- Documentos curados del mes (PDFs en `data/news/`)
- Sin acceso a meses futuros — zero data leakage

---

## Dos modos de ejecución

### Modo histórico (corre UNA sola vez)
- Descarga y almacena toda la serie disponible hasta diciembre 2023
- Llena `data/cache/` con precios, CCL y datos macro
- Una vez generado, este cache es **inmutable**

### Modo live (corre mes a mes desde enero 2024)
- Lee el cache histórico existente
- Agrega solo el mes nuevo al final
- Toma decisiones de allocación con la información disponible hasta ese mes
- Guarda resultados en `results/{año}/`

---

## Fórmula de ponderación dinámica

Los pesos ya **no son equal weight**. Cada mes se calculan dinámicamente con 5 variables:

```
peso_i = señal_técnica_i      × 0.40
       + volatilidad_inversa_i × 0.30
       + correlación_inversa_i × 0.10
       + nlp_empresa_i         × 0.10
       + nlp_macro             × 0.10
```

### Definición de cada variable

**señal_técnica_i (40%)**
`price_to_sma30` del activo i — qué tan por encima de su promedio de 30 semanas está el precio. Mayor valor = más peso.

**volatilidad_inversa_i (30%)**
`1 / realized_vol_12` del activo i — activos más estables reciben más peso. Se normaliza entre 0 y 1 antes de aplicar.

**correlación_inversa_i (10%)**
Qué tan poco correlacionado está el activo con el resto del portfolio ese mes. Activos que se mueven de forma independiente reciben más peso — mejora la diversificación real.

**nlp_empresa_i (10%)**
Score FinBERT de los 6-K presentados por ese ticker en SEC Edgar ese mes. Distinto por cada activo. Rango: -1.0 a +1.0.

**nlp_macro (10%)**
Score FinBERT agregado de los documentos macro del mes: REM, BCRA Monetario Mensual, INDEC Informa, Fed FOMC. El mismo valor para todos los activos ese mes. Rango: -1.0 a +1.0.

### Parámetros configurables
Los pesos (0.40, 0.30, 0.10, 0.10, 0.10) viven en `config/settings.yaml` y son editables sin tocar el código.

---

## Lógica de asignación

### Régimen Risk-On
- Ponderación dinámica sobre **todos los ADRs con señal positiva** ese mes
- ADRs con señal negativa → 0% ese mes

### Régimen Risk-Off
- Ponderación dinámica aplicada únicamente sobre los **top 5 ADRs** por score combinado
- Los 5 activos con peor score reciben 0% ese mes
- Risk-On/Off es una variable de contexto que modifica el número de activos activos, no los pesos relativos entre ellos

### Rebalanceo
- Se dispara **solo cuando cambia el régimen** (Risk-On ↔ Risk-Off)
- Pesos siempre suman 100%
- Sin cash, sin short selling, sin apalancamiento
- Sin renta fija — portfolio 100% equity

---

## Frecuencia operativa

| Elemento | Frecuencia |
|----------|------------|
| Datos de entrada | Semanal |
| Decisiones de allocación | Mensual |
| Rebalanceo | Solo al cambio de régimen |

---

## Señales técnicas

Calculadas sobre el **retorno agregado del portfolio** (promedio del universo activo ese mes) para el clasificador de régimen, y **por activo individual** para la ponderación dinámica:

- SMA 20
- SMA 30
- Momentum (rolling sum 12 semanas)
- Volatilidad realizada (rolling std 12 semanas)

---

## Clasificación de régimen

- **PCA** para reducción de dimensionalidad de las señales técnicas y macro
- **SVM** para clasificación binaria: Risk-On / Risk-Off
- Entrenado en período in-sample: 2015-01 → 2023-12
- Aplicado walk-forward mes a mes desde enero 2024
- El régimen es una variable de contexto — no el único determinante de los pesos

---

## NLP — Sentimiento (dos componentes)

### NLP por empresa — nlp_empresa_i
- Input: 6-K y 20-F de cada ticker en SEC Edgar del mes en curso
- Modelo: FinBERT
- Output: score por ticker, rango [-1.0, +1.0]
- Entra en la fórmula de pesos como variable individual por activo

### NLP macro — nlp_macro
- Input: documentos macro del mes (ver Dataset de documentos abajo)
- Modelo: FinBERT
- Output: score agregado único para todo el portfolio ese mes, rango [-1.0, +1.0]
- Entra en la fórmula de pesos igual para todos los activos

---

## Dataset de documentos — data/news/

993 documentos totales organizados en `data/news/{año}/{mes}/`.
Todos los archivos van en la carpeta del mes en que se **publican** (regla anti-leakage), no del mes que cubren.

### Fuentes automatizadas
| Fuente | Tipo | Cobertura | Archivos |
|--------|------|-----------|---------|
| SEC Edgar (6-K y 20-F) | Por empresa | Ene 2024 → Dic 2025 | 911 |
| BCRA REM | Macro AR | Ene 2024 → Dic 2025 | 24 |
| BCRA Monetario Mensual | Macro AR | Jun 2024 → Dic 2025 | 19 |
| INDEC Informa | Macro AR | Ene 2024 → Dic 2025 | 24 |
| Fed FOMC Minutes | Macro Global | Ene 2024 → Dic 2025 | 15 |

### Limitación documentada — enero a mayo 2024
El BCRA no publicó el Informe Monetario Mensual durante enero-mayo 2024 (gestión Milei). Esos meses tienen REM + SEC filings pero sin Monetario Mensual. Esta limitación se documenta en el paper como restricción de disponibilidad de datos, no como error del sistema.

### Fuentes manuales (agregar cuando estén disponibles)
- IEF BCRA (Informe de Estabilidad Financiera) — semestral, mayo y noviembre
- Reportes FMI sobre Argentina
- Reportes de calificadoras (Fitch, Moody's) sobre Argentina

---

## Benchmarks

| Benchmark | Descripción | Conversión |
|-----------|-------------|------------|
| EW Buy & Hold ADRs | Equal weight estático de los 10 ADRs | Ya en USD |
| Merval en USD | `^MERV` via yfinance | Dividido por CCL (Contado con Liquidación) |
| FIMA Acciones USD | FCI de acciones argentinas | Pendiente — datos no disponibles públicamente sin auth |

**AL30D fue eliminado de los benchmarks** — datos históricos no disponibles en fuentes reguladas.

---

## Stack tecnológico

```
Python 3.11+
pandas / numpy          → manipulación de datos
scikit-learn            → PCA, SVM, correlación
vectorbt                → motor de backtest
transformers / FinBERT  → NLP empresa y macro
matplotlib              → visualización estilo académico
yfinance                → descarga de precios ADRs y Merval
requests / BeautifulSoup → scraping macro
python-dotenv           → variables de entorno
pyyaml                  → configuración
pdfminer.six            → extracción texto de PDFs
```

---

## Configuración del proyecto

```
config/
├── assets.yaml      → lista de tickers, sectores y metadatos
├── periods.yaml     → fechas in-sample y out-of-sample
└── settings.yaml    → parámetros del modelo incluyendo pesos de la fórmula
```

Pesos de la fórmula en settings.yaml:
```yaml
allocation_weights:
  signal_weight: 0.40
  vol_weight: 0.30
  corr_weight: 0.10
  nlp_company_weight: 0.10
  nlp_macro_weight: 0.10
```

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
├   ├── benchmarks/
│   ├── cache/
│   │   ├── prices/          ← ADRs, Merval (cache histórico inmutable)
│   │   ├── macro/           ← CCL, inflación, reservas BCRA
│   │   └── models/          ← regime_pipeline.joblib + .meta.json
│   ├── news/
│   │   ├── 2024/
│   │   │   ├── 01/          ← SEC filings + REM ene 2024
│   │   │   └── ...
│   │   └── 2025/
│   │       └── ...
│   └── exports/
│
├── notebooks/
│
├── results/
│   ├── 2024/
│   └── 2025/
│
├── src/
│   ├── data/                ← descarga, scraping, validación y cache
│   │   └── news_downloader.py ← SEC Edgar + BCRA + INDEC + FOMC
│   ├── features/            ← señales técnicas (SMA, momentum, volatilidad)
│   ├── signals/             ← clasificación de régimen (PCA + SVM)
│   ├── allocation/          ← ponderación dinámica 5 variables
│   ├── backtest/            ← walk-forward, benchmarks
│   ├── metrics/             ← Sharpe, drawdown, Treynor, Jensen, Calmar
│   ├── nlp/                 ← FinBERT empresa + macro
│   └── utils/               ← logging, helpers, snapshots
│
├── tests/
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── CLAUDE.md
```

---

## Pipeline — main.py

```
[MODO HISTÓRICO — solo primera corrida]
1. Descarga 2015→2023, almacena data/cache/ (inmutable)
2. Entrena Pipeline(StandardScaler → PCA≥95% → SVC RBF)
3. Guarda regime_pipeline.joblib

[MODO LIVE — corre mes a mes desde enero 2024]
4. Lee cache histórico + agrega datos del mes
5. Calcula señales técnicas (SMA, momentum, volatilidad) por activo
6. Procesa documentos del mes con FinBERT:
   - nlp_empresa_i por cada ticker (sus 6-K en SEC)
   - nlp_macro del mes (REM + Monetario + INDEC + FOMC)
7. Clasifica régimen (PCA + SVM)
8. Calcula pesos dinámicos con fórmula de 5 variables
9. Registra decisión en log (activos, pesos, régimen, señales, NLP scores)
10. Simulación walk-forward acumulada
11. Calcula métricas vs benchmarks (Sharpe, drawdown, Treynor, Jensen, Calmar)
12. Exporta en results/{año}/
```

---

## Outputs por corrida

En `results/{año}/`:
- `portfolio_weights.csv` — pesos mensuales por activo
- `performance_metrics.csv` — métricas vs benchmarks
- `snapshot_{timestamp}.json` — reproducible completo
- `figures/equity_curves.png/.pdf`
- `figures/drawdown.png/.pdf`
- `logs/run_{timestamp}.log`

---

## Principios de desarrollo (NO negociables)

1. **Sin data leakage** — el modelo solo ve lo que existiría en esa fecha real
2. **El modelo no conoce los labels bull/bear** — son etiquetas analíticas a posteriori
3. **Sin CSVs manuales para precios** — todo por scraping o API automática
4. **Cache histórico inmutable** — una vez generado, no se re-descarga
5. **Reproducibilidad** — toda corrida genera snapshot JSON
6. **Trazabilidad** — toda decisión queda en el log con su señal y scores NLP
7. **Modularidad** — cada módulo de src/ tiene una sola responsabilidad
8. **Portfolio 100% equity** — sin renta fija, sin AL30D
9. **Universo inmutable** — los candidatos no cambian durante el proyecto
10. **Fechas dispares aceptadas** — se documenta en el log

---

## Contexto académico

- El paper compara gestión **humana** vs. **algorítmica**
- Los períodos bull/bear son etiquetas **analíticas a posteriori**
- El informe está dirigido a lectores **no técnicos** con profundidad académica
- Toda decisión de diseño debe poder **justificarse metodológicamente**

### Limitaciones documentadas para el paper
- AL30D: datos históricos no disponibles en fuentes equivalentes a SEC
- EMBI: endpoint de Ámbito removido — feature ausente en el SVM
- BCRA Monetario Mensual: no publicado enero-mayo 2024 bajo gestión Milei
- FIMA Acciones: API requiere autenticación — benchmark pendiente
- FOMC Nov 2025: minuta aún no publicada al momento de la descarga

---

## Estado actual del proyecto

### Completado ✅
- `src/data/` — 8 archivos + news_downloader.py
- `src/features/` — señales técnicas
- `src/signals/` — PCA + SVM
- `src/allocation/` — **pendiente actualizar a ponderación dinámica**
- `src/backtest/` — walk-forward + benchmarks
- `src/metrics/` — métricas + visualización con date_range
- `src/nlp/` — **pendiente separar en nlp_empresa + nlp_macro**
- `src/utils/` — logging y helpers
- `main.py` — pipeline integrado
- `config/` — assets, periods, settings
- `data/news/` — 993 documentos (911 SEC + 82 macro)
- Primera corrida histórica y live completadas
