# CLAUDE.md — gestionHvsA-adrsArgy

> **Para Claude sin contexto:** este archivo es la fuente de verdad del proyecto. Leelo completo antes de tocar cualquier archivo. El proyecto está **terminado** — no hay cambios estructurales pendientes.

---

## Qué es este proyecto

Sistema de **Monthly Tactical Asset Allocation** sobre diez ADRs argentinos denominados en USD.

**Objetivo académico:** comparar el desempeño de gestión algorítmica (este sistema, llamado PGA) frente a gestión humana institucional (FCI Acciones Argentina) en dos períodos de mercado opuestos: uno alcista (2024) y uno bajista (2025).

El sistema toma decisiones mes a mes usando únicamente información disponible hasta el cierre de ese mes. Es un observador que va decidiendo — no un oráculo. **Zero data leakage** es el principio de diseño más importante.

**Audiencia del paper final:** estudiantes de finanzas y lectores no técnicos.

---

## Resultados finales — para referencia rápida

| Período | PGA | FCI Acciones AR | SPY | EEM |
|---------|-----|-----------------|-----|-----|
| 2024 (bull) | **+130.2%** | +111.8% | +25.0% | +6.7% |
| 2025 (bear) | -22.1% | -14.6% | +17.8% | **+34.3%** |
| **2024–2025 acumulado** | **+79.4%** | +74.6% | +44.3% | +47.3% |

**Sharpe 2024:** PGA 2.68 vs FCI 1.85 — el algoritmo fue más eficiente en el bull market.
**Sharpe 2025:** PGA -0.30 vs FCI 0.08 — el FCI fue más eficiente en el bear market.
**Hallazgo central:** octubre 2025 — IRS al 100%, PGA +30.9% vs EW +67.6% (-36.7pp de diferencia en un mes).

---

## Universo de activos (inmutable)

10 ADRs argentinos cotizando en NYSE/NASDAQ, denominados en USD:

| Ticker | Empresa | Sector |
|--------|---------|--------|
| YPF | YPF S.A. | Energía petrolera |
| GGAL | Grupo Financiero Galicia | Bancario |
| BMA | Banco Macro | Bancario |
| PAM | Pampa Energía | Energía eléctrica |
| TGS | Transportadora de Gas del Sur | Infraestructura energética |
| CEPU | Central Puerto | Generación eléctrica |
| EDN | Empresa Distribuidora Norte (Edenor) | Distribución eléctrica |
| LOMA | Loma Negra | Construcción / Materiales |
| CRESY | Cresud | Agro / Real Estate |
| IRS | IRSA Inversiones y Representaciones | Real Estate |

**Reglas del universo:**
- El universo es fijo durante todo el proyecto — no se agregan ni quitan tickers.
- Cada mes el modelo puede asignar 0% a cualquier ADR.
- Un activo con 0% un mes puede reincorporarse el mes siguiente.
- No hay renta fija, no hay cash, no hay short selling, no hay apalancamiento.
- AL30D fue descartado: datos históricos no disponibles en fuentes reguladas equivalentes a SEC.

---

## Períodos

```
2015-01 → 2023-12   IN-SAMPLE
                     Cache histórico, entrenamiento de PCA + SVM.
                     Inmutable. No se reporta como desempeño.

2024-01 → 2024-12   OUT-OF-SAMPLE — Bull market (etiqueta a posteriori)
2025-01 → 2025-12   OUT-OF-SAMPLE — Bear market (etiqueta a posteriori)
```

---

## Arquitectura del sistema

### Pipeline mensual (main.py)

```
MODO HISTÓRICO (primera corrida, una sola vez):
1. Descarga precios 2015→2023 via yfinance → data/cache/prices/adrs.parquet
2. Descarga CCL y macro → data/cache/macro/
3. Entrena StandardScaler → PCA(≥95% varianza) → SVC(kernel RBF)
4. Guarda data/cache/models/regime_pipeline.joblib

MODO LIVE (corre mes a mes desde enero 2024):
python main.py --mode live --month 2025-12

5. Lee cache histórico + agrega datos del mes nuevo
6. Calcula señales técnicas por activo (SMA20, SMA30, momentum, vol)
7. Carga documentos de data/news/{año}/{mes}/ y procesa con FinBERT:
   → nlp_empresa_i: score por ticker (sus 6-K en SEC)
   → nlp_macro: score único del mes (REM + Monetario + INDEC + FOMC)
8. Clasifica régimen: PCA + SVM → "risk_on" o "risk_off"
9. Calcula pesos dinámicos con fórmula de 5 variables
10. Rebalanceo solo si cambió el régimen vs. mes anterior
11. Simulación walk-forward acumulada
12. Exporta results/{año}/
```

### Fórmula de ponderación dinámica

```
peso_i = señal_técnica_i      × 0.40   (price_to_sma30 — tendencia del precio)
       + volatilidad_inversa_i × 0.30   (1/vol_12 — premia activos estables)
       + correlación_inversa_i × 0.10   (1 - corr_promedio — premia diversificación)
       + nlp_empresa_i         × 0.10   (FinBERT sobre 6-K del ticker ese mes)
       + nlp_macro             × 0.10   (FinBERT sobre docs macro del mes)
```

Los coeficientes viven en `config/settings.yaml` → `allocation_weights`. Son editables sin tocar código.
Todos los sub-componentes se normalizan a [0,1] con min-max antes de aplicar la fórmula.
Los scores NLP en [-1,1] se mapean a [0,1] antes de ponderar.

### Lógica de asignación

**Risk-On:** ponderación dinámica sobre todos los ADRs con `price_to_sma30 > 1.0`.
ADRs con señal negativa → 0% ese mes.

**Risk-Off:** igual que Risk-On pero restringido al **top 5** por score compuesto.
Los 5 restantes → 0% ese mes.

**Fallback:** si ningún ADR pasa el filtro → equal weight sobre el universo completo + warning en log.

**Rebalanceo:** solo cuando cambia el régimen (Risk-On ↔ Risk-Off). Entre cambios, las posiciones se mantienen y los pesos derivan con los precios.

---

## Benchmarks

| Benchmark | Descripción | Fuente |
|-----------|-------------|--------|
| FCI Acciones Argentina — USD | FCI de renta variable argentina, convertido a USD con CCL | CSV manual: `data/benchmarks/fima_benchmark_usd.csv` |
| SPY | SPDR S&P 500 ETF Trust | yfinance |
| EEM | iShares MSCI Emerging Markets ETF | yfinance |

**Nota sobre el FCI:** el nombre del fondo no se divulga a pedido del director del trabajo. Los datos de VCP en ARS fueron provistos por un integrante del equipo y convertidos a USD usando el CCL del día de cierre mensual (`vcp_date`). El CCL usado es el mismo del cache del proyecto para garantizar consistencia metodológica.

---

## Dataset de documentos — data/news/

**Total: 993 documentos** en `data/news/{año}/{mes}/`.

**Regla anti-leakage:** cada archivo va en la carpeta del mes en que se **publica**, no del mes que cubre.

### Fuentes automatizadas (news_downloader.py)

| Fuente | Función | Cobertura | Archivos |
|--------|---------|-----------|---------|
| SEC Edgar (6-K y 20-F) | NLP empresa por ticker | Ene 2024 → Dic 2025 | 911 |
| BCRA REM | NLP macro | Ene 2024 → Dic 2025 | 24 |
| BCRA Monetario Mensual | NLP macro | **Jun 2024 → Dic 2025** | 19 |
| INDEC Informa | NLP macro | Ene 2024 → Dic 2025 | 24 |
| Fed FOMC Minutes | NLP macro | Ene 2024 → Dic 2025 | 15 |

**Limitación documentada:** el BCRA no publicó el Informe Monetario Mensual durante enero–mayo 2024 (gestión Milei). Esos 5 meses operan solo con REM + SEC filings.

**FOMC nov-2025:** minuta no publicada al momento de la descarga → 404 esperado. La función `download_macro_documents()` es idempotente: re-ejecutarla descargará el faltante cuando esté disponible.

### Archivos manuales (data/benchmarks/)

- `fima_acciones_monthly_composition.xlsx` — dos pestañas: `ponderaciones_mensuales` y `returns`
- `fima_benchmark_usd.csv` — serie mensual FCI en USD con CCL (generada a partir del xlsx)

---

## NLP — FinBERT

**Modelo:** `ProsusAI/finbert` — entrenado en textos financieros en inglés.
**Score:** P(positive) - P(negative) → rango [-1.0, +1.0].
**Chunking:** 450 tokens con overlap de 50 tokens → score = media de chunks.
**Límite:** primeros 5.000 caracteres por documento (para velocidad en CPU).
**Partición por nombre de archivo:**
- Contiene un ticker conocido → `nlp_empresa_i` de ese ticker.
- Resto (bcra_*, indec_*, fed_fomc_*) → `nlp_macro` del mes.

---

## Clasificador de régimen

```
Pipeline: StandardScaler → PCA (≥95% varianza) → SVC (kernel RBF, probability=True)
Entrenado en: 2015-01 → 2023-12 (in-sample)
Aplicado: walk-forward mes a mes en 2024-2025
Output: {"regime": "risk_on" | "risk_off", "probability": float}
```

El régimen es **una variable de contexto** que modifica el número de activos activos — no determina los pesos relativos entre activos activos. Valores de probabilidad cercanos a 0.5 indican ambigüedad.

---

## Estructura de archivos

```
gestionHvsA-adrsArgy/
│
├── CLAUDE.md                    ← este archivo
├── README.md
├── main.py                      ← punto de entrada
├── requirements.txt
├── .env                         ← no versionar
├── .gitignore
│
├── config/
│   ├── assets.yaml
│   ├── periods.yaml
│   └── settings.yaml            ← allocation_weights aquí
│
├── data/
│   ├── benchmarks/
│   │   ├── fima_acciones_monthly_composition.xlsx
│   │   └── fima_benchmark_usd.csv
│   ├── cache/                   ← en .gitignore
│   │   ├── prices/adrs.parquet
│   │   ├── macro/ccl.parquet
│   │   └── models/regime_pipeline.joblib
│   ├── news/                    ← en .gitignore (993 docs)
│   │   ├── 2024/{01..12}/
│   │   └── 2025/{01..12}/
│   └── exports/
│
├── src/
│   ├── data/
│   │   ├── downloader.py
│   │   ├── scraper_ccl.py
│   │   ├── scraper_macro.py
│   │   ├── cache_manager.py
│   │   ├── validator.py
│   │   ├── loader.py
│   │   └── news_downloader.py
│   │       download_news_for_period(start, end) → SEC filings
│   │       download_macro_documents(start, end) → BCRA + INDEC + FOMC
│   │
│   ├── features/
│   │   └── features.py
│   │       Señales: price_to_sma20, price_to_sma30, momentum_12w,
│   │                realized_vol_12, price
│   │       Output: FeatureMatrix con MultiIndex (ticker, signal_name)
│   │
│   ├── signals/
│   │   └── regime.py
│   │       predict_regime(features) → {"regime": str, "probability": float}
│   │
│   ├── allocation/
│   │   └── allocator.py
│   │       compute_weights(regime_result, assets_features, nlp_result)
│   │         → AllocationResult(weights, active_adrs, excluded_adrs, regime, probability)
│   │       Lee pesos desde config/settings.yaml
│   │       Risk-Off: top_5 por score compuesto
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   │   run_backtest(monthly_allocations, adrs, merval, ccl)
│   │   │     → BacktestResult(portfolio, benchmarks, rebalance_dates)
│   │   │   Execution lag: pesos de mes M se ejecutan en primer día hábil de M+1
│   │   └── benchmarks.py
│   │       compute_benchmarks(adrs, merval, ccl, start, end)
│   │         → dict: "ew_bnh", "merval_usd", "fima_acciones_usd"
│   │
│   ├── metrics/
│   │   ├── metrics.py
│   │   └── visualizer.py
│   │       _apply_chart_style(ax, title) → estilo base unificado
│   │       plot_equity_curves(result, output_dir, date_range=None)
│   │       plot_drawdown(result, output_dir, date_range=None)
│   │
│   ├── nlp/
│   │   └── nlp.py
│   │       compute_monthly_sentiment(year, month, base_dir, _pipeline)
│   │         → {"macro_score": float, "company_scores": {ticker: float}}
│   │
│   └── utils/
│       └── utils.py
│
├── results/
│   ├── allocation_complete_table.csv   ← tabla maestra 24×28
│   ├── 2024/
│   │   ├── portfolio_weights.csv
│   │   ├── performance_metrics.csv
│   │   └── figures/
│   ├── 2025/
│   │   ├── portfolio_weights.csv
│   │   ├── performance_metrics.csv
│   │   └── figures/
│   └── figures/                 ← gráficos finales del paper
│       ├── equity/
│       │   ├── PGA_equity_curve.png
│       │   ├── PGA_equity_2024.png
│       │   ├── PGA_equity_2025.png
│       │   ├── FCI_equity_curve.png
│       │   ├── SPY_EEM_equity_curve.png
│       │   ├── PGA_vs_FIMA.png
│       │   └── PGA_benchmarks_completo.png
│       ├── heatmap/
│       │   ├── PGA_weights_heatmap.png
│       │   └── FCI_weights_heatmap.png
│       ├── metrics/
│       │   ├── metricas_comparativas.csv
│       │   └── retornos_mensuales_PGA_FCI.csv
│       └── composition/
│           ├── PGA_composition_2024.png
│           ├── PGA_composition_2025.png
│           ├── FCI_composition_2024.png
│           └── FCI_composition_2025.png
│
└── tests/
```

---

## Estilo de gráficos — especificación definitiva

Todos los gráficos usan `_apply_chart_style()` de `src/metrics/visualizer.py`.

**Paleta (no cambiar):**
- PGA = `#1a3a6b` (azul oscuro)
- FCI Acciones Argentina = `#006d6d` (teal)
- SPY = `#8b1a1a` (rojo vino)
- EEM = `#2e6b3e` (verde oscuro)
- Valores negativos en anotaciones = `#8b0000`

**Reglas:**
- Tipografía Arial en todo
- Fondo blanco, grilla solo horizontal `#e0e0e0`
- Bandas sombreadas: 2024=`#1a3a6b` alpha=0.06, 2025=`#e8751a` alpha=0.06
- Etiquetas "2024" y "2025" en gris `#888888` en la parte superior
- Línea vertical sólida gris en 2025-01-01
- Anotaciones finales: "+XX.X% (desde ene 2024)" y "-XX.X% (desde ene 2025)"
- Solo PNG (PDF al final, de una vez)

---

## allocation_complete_table.csv — columnas

```
mes, regimen, prob_svm, nlp_macro, activos_activos,
YPF, GGAL, BMA, PAM, TGS, CEPU, EDN, LOMA, CRESY, IRS,   ← pesos (%)
YPF_delta, GGAL_delta, ..., IRS_delta,                      ← cambio vs mes anterior (pp)
ret_portfolio_%, ret_ew_bnh_%, diff_vs_ew_pp
```

Nota: `ret_portfolio_%` de enero 2024 es "—" (mes de inicialización).

---

## Comandos frecuentes

```bash
# Activar entorno
source .venv/Scripts/activate          # Git Bash
.venv\Scripts\activate                 # PowerShell

# Pipeline completo
python main.py --mode live --month 2025-12

# Test rápido (3 meses)
python main.py --mode live --month 2024-03

# Descargar documentos SEC
python -c "from src.data.news_downloader import download_news_for_period; print(download_news_for_period('2024-01', '2025-12'))"

# Descargar documentos macro
python -c "from src.data.news_downloader import download_macro_documents; print(download_macro_documents('2024-01', '2025-12'))"

# Verificar imports
python -c "from src.allocation import compute_weights; from src.nlp import compute_monthly_sentiment; from src.backtest import run_backtest; print('OK')"
```

---

## Limitaciones documentadas

1. **Concentración extrema:** filtro `price_to_sma30 > 1.0` puede dejar 1-2 activos activos en mercados bajistas. No hay mínimo de activos ni cap de peso.
2. **AL30D descartado:** sin instrumento defensivo en Risk-Off. Sistema 100% equity.
3. **BCRA Monetario ene-may 2024:** no publicado. NLP macro esos meses: solo REM + FOMC.
4. **FOMC nov-2025:** minuta no publicada al momento de la descarga.
5. **FIMA API bloqueada:** datos cargados manualmente. fima_acciones_usd() lee CSV local.
6. **Muestra out-of-sample:** 24 meses = suficiente para descripción, insuficiente para significancia estadística.
7. **Sesgo de supervivencia:** los 10 ADRs continúan cotizando al cierre del período.
8. **FinBERT en CPU:** lento sin GPU. Límite de 5.000 chars/doc mitiga el problema.

---

## Estado del proyecto

**TERMINADO.** 24/24 meses exitosos, 0 errores. Paper redactado. Gráficos aprobados.

Extensiones posibles (no implementadas):
- Floor de activos activos (mínimo 3-5 para evitar concentración extrema).
- Cap de peso máximo por activo (25-30%).
- Instrumento de refugio en USD para Risk-Off.
- Calibración de pesos de la fórmula mediante optimización in-sample.