# Hierarchical graph attention network for miRNA-disease associations prediction

## Dependecies
- Python 3.7
- dgl 0.5.1
- numpy 1.16.5
- torch 1.9.0
- pandas 0.25.1

## Dataset
miRNA-disease associations,
miRNA-lncRNA associations,
disease-lncRNA associations,
miRNA features,
disease features,
lncRNA features,

###### Model options
```
--epochs           int     Number of training epochs.                 Default is 1000.
--attn_size        int     Dimension of attention.                    Default is 64.
--attn_heads       int     Number of attention heads.                 Default is 8.
--out_dim          int     Output dimension after feature extraction  Default is 64.
--dropout          float   Dropout rate                               Default is 0.5.
--slope            float   Slope                                      Default is 0.2.
--lr               float   Learning rate                              Default is 0.001.
--wd               float   weight decay                               Default is 5e-3.

```

###### How to run?
```
Run main.py

```
