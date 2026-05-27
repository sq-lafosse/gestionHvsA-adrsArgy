# gestionHvsA-adrsArgy

**Sistema de gestión algorítmica de carteras sobre ADRs argentinos en USD**

Sistema de _Monthly Tactical Asset Allocation_ que invierte mensualmente en diez ADRs argentinos cotizados en NYSE/NASDAQ. Combina señales técnicas de precio, un clasificador de régimen de mercado (PCA + SVM) y análisis de sentimiento financiero (FinBERT) para decidir cada mes qué activos incluir y con qué peso. Desarrollado como trabajo de investigación académica para comparar gestión algorítmica (PGA) frente a gestión humana institucional (FCI Acciones Argentina).

---

## 📚 Contexto académico

Este repositorio es la implementación completa del sistema PGA (_Portfolio Gestión Algorítmica_), desarrollado como trabajo de investigación en el marco de la carrera de Economía / Finanzas de la UADE.

El estudio evalúa si un sistema de reglas cuantitativas puede superar a un fondo de inversión administrado por gestores profesionales, usando el mismo universo de activos (ADRs argentinos en USD) durante dos períodos de mercado opuestos:

- **2024** — mercado alcista (_bull market_): los ADRs argentinos subieron fuertemente impulsados por el programa económico de la nueva administración.
- **2025** — mercado bajista (_bear market_): corrección pronunciada en activos argentinos en el primer semestre, con recuperación parcial en el segundo.

El paper completo con metodología, resultados y análisis está disponible en el repositorio.

---

## 📊 Resultados principales

| Cartera | 2024 (bull) | 2025 (bear) | Acumulado 2024–2025 |
|---------|:-----------:|:-----------:|:-------------------:|
| **PGA** (este sistema) | **+130.2%** | -22.1% | **+79.4%** |
| FCI Acciones Argentina | +111.8% | -14.6% | +74.6% |
| SPY (S&P 500) | +25.0% | +17.8% | +44.3% |
| EEM (Emergentes) | +6.7% | **+34.3%** | +47.3% |

**Sharpe 2024:** PGA 2.68 vs FCI 1.85 — el algoritmo fue más eficiente en el bull market.
**Sharpe 2025:** PGA -0.30 vs FCI 0.08 — el FCI fue más eficiente en el bear market.

El sistema no tiene acceso a información del futuro. Cada decisión mensual usa únicamente datos disponibles hasta el cierre de ese mes (_zero data leakage_).

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│  Datos históricos (2015–2023)                                   │
│  Precios ADRs + CCL + variables macro                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ (una sola vez)
                     ▼
           ┌──────────────────┐
           │  Cache local     │  data/cache/prices/
           │  + Modelo SVM    │  data/cache/models/
           └────────┬─────────┘
                    │
   ┌────────────────▼──────────────────────────────────┐
   │  Pipeline mensual (enero 2024 → diciembre 2025)   │
   │                                                   │
   │  1. Señales técnicas por activo                   │
   │     SMA20, SMA30, momentum, volatilidad           │
   │                                                   │
   │  2. NLP — FinBERT                                 │
   │     → nlp_empresa_i  (6-K de cada ticker en SEC) │
   │     → nlp_macro      (BCRA + INDEC + FOMC)        │
   │                                                   │
   │  3. Régimen de mercado                            │
   │     StandardScaler → PCA → SVC                   │
   │     → "risk_on" | "risk_off"                      │
   │                                                   │
   │  4. Ponderación dinámica                          │
   │     peso_i = técnica×0.40 + vol_inv×0.30          │
   │            + corr_inv×0.10 + nlp_emp×0.10         │
   │            + nlp_macro×0.10                       │
   │                                                   │
   │  5. Rebalanceo (solo si cambia el régimen)        │
   └────────────────┬──────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Backtest            │
         │  walk-forward acum.  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Resultados          │
         │  results/figures/    │
         │  results/20XX/       │
         └──────────────────────┘
```

---

## ⚙️ Requisitos

- **Python 3.11+**
- Dependencias principales:

| Paquete | Versión mínima | Uso |
|---------|---------------|-----|
| pandas | 2.0 | Procesamiento de datos |
| numpy | 1.24 | Cálculo numérico |
| scikit-learn | 1.3 | PCA + SVM (clasificador de régimen) |
| yfinance | 0.2.36 | Descarga de precios históricos |
| transformers | 4.35 | FinBERT (NLP) |
| torch | 2.0 | Backend de FinBERT (CPU-only) |
| matplotlib | 3.7 | Visualizaciones |
| pyyaml | 6.0 | Configuración |
| beautifulsoup4 | 4.12 | Scraping de documentos |
| pdfminer.six | 20221105 | Extracción de texto de PDFs |

> **Nota sobre PyTorch:** el sistema corre completamente en CPU. No se requiere GPU ni CUDA. El modelo FinBERT (~440 MB) se descarga automáticamente en la primera ejecución.

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/sq-lafosse/gestionHvsA-adrsArgy.git
cd gestionHvsA-adrsArgy
```

### 2. Crear el entorno virtual

```bash
# Windows — PowerShell
python -m venv .venv
.venv\Scripts\activate

# Windows — Git Bash
python -m venv .venv
source .venv/Scripts/activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar PyTorch (CPU-only)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> Si `torch` ya quedó instalado en el paso anterior desde PyPI (con CUDA), reemplazarlo con el comando de arriba para obtener la versión CPU más liviana.

### 5. Variables de entorno (opcional)

El sistema no requiere claves API. Si necesitás configurar rutas personalizadas, podés crear un archivo `.env` en la raíz del proyecto. Para la ejecución estándar no es necesario.

---

## ▶️ Cómo correr el sistema

### Paso 1 — Primera vez: generar el cache histórico

```bash
python main.py --mode historical
```

Descarga los precios de los 10 ADRs desde 2015 hasta 2023, el tipo de cambio CCL y las variables macro, y entrena el clasificador de régimen (PCA + SVM).

**Duración estimada:** 5–10 minutos (depende de la velocidad de descarga).
**Frecuencia:** solo una vez. Las siguientes ejecuciones detectan el cache existente y lo saltan.

### Paso 2 — Descargar los documentos para el análisis NLP

Los documentos (filings SEC, informes BCRA, INDEC, FOMC) se descargan así:

```bash
# Filings SEC (6-K y 20-F) de los 10 ADRs — ~911 documentos
python -c "from src.data.news_downloader import download_news_for_period; download_news_for_period('2024-01', '2025-12')"

# Documentos macro (BCRA + INDEC + FOMC) — ~82 documentos
python -c "from src.data.news_downloader import download_macro_documents; download_macro_documents('2024-01', '2025-12')"
```

**Duración estimada:** 10–30 minutos (993 documentos en total).
**Nota:** las funciones son idempotentes — re-ejecutarlas descarga solo lo que falta.

### Paso 3 — Correr el pipeline completo

```bash
python main.py --mode live --month 2025-12
```

Procesa los 24 meses del período de evaluación (enero 2024 → diciembre 2025), mes a mes en secuencia. Para cada mes: actualiza el cache de precios, calcula señales técnicas, procesa documentos con FinBERT, clasifica el régimen, calcula los pesos de la cartera y ejecuta el backtest walk-forward acumulado.

**Duración estimada:** 60–90 minutos en CPU (el cuello de botella es FinBERT).

**Tip para pruebas rápidas:** para correr solo los primeros 3 meses:

```bash
python main.py --mode live --month 2024-03
```

### Paso 4 — Ver los resultados

| Ruta | Contenido |
|------|-----------|
| `results/allocation_complete_table.csv` | Tabla maestra: régimen, pesos y retornos de los 24 meses |
| `results/figures/equity/` | Curvas de retorno comparativas (PGA vs benchmarks) |
| `results/figures/heatmap/` | Mapas de calor de composición de cartera |
| `results/figures/composition/` | Tortas de composición promedio por período |
| `results/figures/metrics/metricas_comparativas.csv` | Tabla de métricas (Sharpe, MaxDD, Calmar, etc.) |
| `results/2024/` | Métricas y gráficos específicos de 2024 |
| `results/2025/` | Métricas y gráficos específicos de 2025 |

---

## 📁 Estructura de carpetas

```
gestionHvsA-adrsArgy/
│
├── main.py                          ← punto de entrada del pipeline
├── requirements.txt
├── CLAUDE.md                        ← fuente de verdad técnica del proyecto
│
├── config/
│   ├── assets.yaml                  ← lista de los 10 ADRs
│   ├── periods.yaml                 ← períodos in-sample / live
│   └── settings.yaml                ← coeficientes de la fórmula y parámetros
│
├── data/
│   ├── benchmarks/
│   │   ├── fima_benchmark_usd.csv   ← serie mensual del FCI en USD (manual)
│   │   └── fima_acciones_monthly_composition.xlsx
│   ├── cache/                       ← generado automáticamente (.gitignore)
│   │   ├── prices/adrs.parquet      ← precios históricos 2015–2025
│   │   ├── macro/ccl.parquet        ← tipo de cambio CCL
│   │   └── models/regime_pipeline.joblib
│   └── news/                        ← documentos NLP (.gitignore, 993 archivos)
│       ├── 2024/{01..12}/
│       └── 2025/{01..12}/
│
├── src/
│   ├── data/          ← descarga, cache y carga de datos
│   ├── features/      ← cálculo de señales técnicas
│   ├── signals/       ← clasificador de régimen (PCA + SVM)
│   ├── allocation/    ← fórmula de ponderación dinámica
│   ├── backtest/      ← motor de simulación y benchmarks
│   ├── metrics/       ← cálculo de métricas y visualizaciones
│   ├── nlp/           ← pipeline FinBERT
│   └── utils/
│
├── results/
│   ├── allocation_complete_table.csv
│   ├── 2024/
│   ├── 2025/
│   └── figures/
│       ├── equity/
│       ├── heatmap/
│       ├── metrics/
│       └── composition/
│
└── tests/
```

---

## 📄 Dataset de documentos NLP

Los 993 documentos utilizados para el análisis de sentimiento **no están incluidos en el repositorio** (están en `.gitignore` por su tamaño), pero se pueden regenerar completamente con los scripts de descarga del Paso 2.

| Fuente | Tipo | Documentos |
|--------|------|-----------|
| SEC Edgar (6-K, 20-F) | Reportes de cada ADR | 911 |
| BCRA — Informe Monetario Mensual | Macro argentina | 19 |
| BCRA — REM | Relevamiento de expectativas | 24 |
| INDEC Informa | Estadísticas oficiales | 24 |
| Fed FOMC Minutes | Política monetaria EE.UU. | 15 |

**Regla anti-leakage:** cada documento va en la carpeta del mes en que se publica, no del mes que cubre. Esto garantiza que el modelo nunca usa información del futuro.

---

## ❓ Preguntas frecuentes

**¿Necesito GPU para correr el sistema?**
No. FinBERT corre completamente en CPU. Es más lento (~2–3 minutos por mes) pero no requiere ningún hardware especial.

**¿Por qué tarda tanto la primera corrida?**
El primer mes descarga el modelo FinBERT desde HuggingFace (~440 MB). Los siguientes meses son más rápidos porque el modelo queda en cache local.

**¿Puedo correr solo algunos meses para probar?**
Sí: `python main.py --mode live --month 2024-03` corre solo enero, febrero y marzo de 2024.

**¿Los resultados van a ser exactamente iguales a los del paper?**
Sí, siempre que uses el mismo cache histórico y los mismos documentos NLP. El sistema es completamente determinístico dado el mismo input.

**¿Dónde están los datos del FCI Acciones Argentina?**
En `data/benchmarks/fima_benchmark_usd.csv`. Son datos provistos manualmente por el equipo de investigación (la API del fondo está bloqueada para acceso público). El nombre del fondo no se divulga a pedido del director del trabajo.

**¿Puedo cambiar los coeficientes de la fórmula de ponderación?**
Sí. Los pesos (0.40, 0.30, 0.10, 0.10, 0.10) viven en `config/settings.yaml` bajo la clave `allocation_weights`. Son editables sin tocar el código fuente.

---

## 📖 Cita

Si usás este trabajo como referencia:

```
Lafosse, S. (2026). gestionHvsA-adrsArgy: Sistema de gestión algorítmica
de carteras sobre ADRs argentinos en USD. GitHub.
https://github.com/sq-lafosse/gestionHvsA-adrsArgy
```

---

## 📝 Licencia

Uso académico. Trabajo de investigación — UADE, 2026.
