Status: Archive (code is provided as-is, no updates expected)

## Mapping Slums with Deep Learning Feature Extraction

This repo contains the code to reproduce the results obtained on the paper "Mapping Slums with Deep Learning Feature Extraction" to be presented at the CDCEO 2022: 2nd Workshop on Complex Data Challenges in Earth Observation (IJCAI-ECAI 2022). 

## Files

`baseline.py` trains a supervised model for each location.<br>
`generating_features.py` generates features (unsupervised learning) for each location.<br>
`creating_raster_complexity.py` creates a raster file with the topological complexity of the areas of interest.<br>
`results_analysis.py` analyses the results obtained using unsupervised learning.<br>
`visualisations_paper.py` produces the graphs shown in the paper.<br>
`useful_functions.py` contains functions that are used multiple times in different scripts.<br>

## Data sets 

The Sentinel-2 data used in this study is available at: [https://frontierdevelopmentlab.github.io/informal-settlements/](https://frontierdevelopmentlab.github.io/informal-settlements/). The other data used in the paper (Topological Analysis of Crowdsourced Digital Maps) can be accessed [here](https://docs.google.com/forms/d/e/1FAIpQLSfao44uX3l8S0qSsaGEb7ufpiY2F2wfhDs8NmrkzlokWqV-ZQ/viewform).



