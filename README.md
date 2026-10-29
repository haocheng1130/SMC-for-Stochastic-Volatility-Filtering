# SMC for Stochastic Volatility Filtering

Final project for CAAM 31511 *Monte Carlo Simulation* (University of Chicago, Spring 2026), on Chapter 9 of the
lecture notes: particle filters for the daily volatility of the S&P 500.

- Report: [`report/CAAM31511_Final_Project.pdf`](report/CAAM31511_Final_Project.pdf) (LaTeX sources in `report/`)
- Notebook: [`particle_filters_stochastic_volatility_filtering.ipynb`](particle_filters_stochastic_volatility_filtering.ipynb)
- Interactive page: [`docs/index.html`](docs/index.html) runs the particle filters in the browser (open the file
  locally; it needs no server)

## What it does

A stochastic volatility (SV) model treats the log-variance $X_j$ of day $j$ as a hidden AR(1) process,
$R_j = a + e^{X_j/2}\eta_j$, $X_{j+1} = \mu + \phi(X_j - \mu) + \sigma_X\xi_{j+1}$. Three versions differ in how the
return shock $\eta$ and the volatility shock $\xi$ are correlated:

| model | correlation | meaning |
|---|---|---|
| (i) | none | volatility moves on its own |
| (ii) | corr($\eta_j$, $\xi_{j+1}$) = $\rho$ | leverage: a fall today raises tomorrow's volatility |
| (iii) | corr($\eta_j$, $\xi_j$) = $\rho$ | today's return and today's volatility move together |

Parameters are estimated on 2000–2015 by maximizing a particle-filter estimate of the likelihood, and the models
are compared by one-step forecasts on 2016-01 to 2021-05 (1343 days), with GARCH(1,1) and GJR-GARCH(1,1) as
benchmarks. The 5-minute realized variance is used only for evaluation.

## Results

**Particle filtering.** In model (iii) the day's return is informative about that day's volatility shock. The
bootstrap filter ignores this: its mean relative ESS is 0.52. A guided proposal, the optimal-filter kernel of the
model with $\sigma_X = 0$, raises it to 0.92 at the same cost per particle; the guided filter with 500 particles
estimates the log-likelihood as precisely as the bootstrap filter with 5000. The noisier bootstrap likelihood also
biases the simulated maximum likelihood estimate of $\rho$ (−0.80 instead of −0.84), because its downward bias grows
with $|\rho|$.

**Held-out forecasts, 2016-01-04 to 2021-05-18:**

| model | predictive log-likelihood | gain of (iii) | $u < 1\%$ | $u > 99\%$ | QLIKE |
|---|---|---|---|---|---|
| (i) no correlation | 4886.4 | 22.0 (t = 2.0) | 1.94% | 0.45% | 0.319 |
| (ii) next-day correlation | 4880.7 | 27.8 (t = 3.1) | 1.86% | 0.22% | 0.334 |
| (iii) same-day correlation | **4908.5** | | 1.41% | 1.04% | 0.344 |
| GARCH(1,1) | 4798.0 | 110.5 (t = 5.7) | 2.08% | 0.37% | 0.363 |
| GJR-GARCH(1,1) | 4803.5 | 105.0 (t = 5.5) | 1.56% | 0.22% | 0.362 |

$u$ is the probability integral transform of the day's return (nominal 1% in each tail). Model (iii) gives the best
forecast distributions and calibrated 1% tails; for the variance level alone (QLIKE against realized variance) the
uncorrelated model is better.

## Repository structure

```
├── sv_tools.py                 particle filters (bootstrap and guided), simulated likelihood, GARCH
├── particle_filters_stochastic_volatility_filtering.ipynb
├── spx_real_data_extension_cleaned.zip   data (S&P 500 returns and realized variance, 2000–2021)
├── outputs/filter_series.csv   daily filter and forecast output of the notebook
├── report/                     LaTeX sources, figures and results.json written by the notebook, compiled PDF
└── docs/                       interactive page: template.html + build_page.py -> index.html
```

## Reproducing

```bash
conda create -n caam31511 python=3.12
conda activate caam31511
pip install -r requirements.txt
python -m ipykernel install --user --name caam31511

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    particle_filters_stochastic_volatility_filtering.ipynb   # about 35 minutes, three processes
python docs/build_page.py
cd report && latexmk -pdf final_merged.tex && cp final_merged.pdf CAAM31511_Final_Project.pdf
```

All randomness is seeded. The report template (`report/final_merged.tex`) is the course template.

## Data

`spx_real_data_extension_cleaned.zip` holds S&P 500 open-to-close log returns and 5-minute realized measures,
2000-01-03 to 2021-05-18 (5358 days), derived from the Oxford-Man Institute's Realized Library (Heber, Lunde,
Shephard and Sheppard, 2009) through a public copy of that dataset; see the README inside the zip.
