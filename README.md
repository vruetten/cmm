# Coherence Mixture Model

[![License](https://img.shields.io/pypi/l/cmm.svg?color=green)](https://github.com/vrutten/cmm/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cmm.svg?color=green)](https://pypi.org/project/cmm)
[![Python Version](https://img.shields.io/pypi/pyversions/cmm.svg?color=green)](https://python.org)
[![CI](https://github.com/vrutten/cmm/actions/workflows/ci.yml/badge.svg)](https://github.com/vrutten/cmm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vrutten/cmm/branch/main/graph/badge.svg)](https://codecov.io/gh/vrutten/cmm)

or for now k-Means Clustering with coherence as distance metric

Unsupervised clustering algoirthm which clusters timeseries based on coherence.


Coherence between two signals is defined as:
```math
 \mathcal{C}(\bm{\hat{x}}(\omega), \bm{\hat{\mu}}(\omega)) = \frac{|S_{x,\mu}(\omega)|^2}{S_{xx}(\omega)\cdot S_{\mu, \mu}(\omega)} =  \frac{|\sum_{k = 1}^K \hat{x}_k(\omega)\cdot \overline{\hat{\mu}_k}(\omega)|^2}{\left(\sum_{k = 1}^K \hat{\mu}_k(\omega)\cdot \overline{\hat{\mu}_k}(\omega)\right)\left(\sum_{k = 1}^K \hat{x}_k(\omega)\cdot \overline{\hat{x}_k(\omega)}\right)}
 ```


``` demo code

xnt: data
k: number of clusters
fs: sampling frequency
nperseg: number of timepoints in each trial (to compute FFT over and average)
itemax: max number of iterations to optimize

```

```
from cmm import CMM

cm = CMM(
    xnt,
    k=k,
    fs=fs,
    nperseg=nperseg,
)

cm.optimize(itemax=itemax)
print(cm.labels)
```
