# Blind Image Quality Assessment via Transformer Predicted Error Map and Perceptual Quality Token

 ![archi](./images/archi.png)





## Environment
 ![](https://img.shields.io/badge/python-3.8-orange.svg) ![](https://img.shields.io/badge/pytorch-1.11.0-green.svg)

> $ pip install -r  requirements.txt 
> 
> $ conda env create -f environment.yaml



## Datasets

In this work we use 6 datasets ([LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23), [TID2013](http://www.ponomarenko.info/tid2013.htm), [KADID10K](http://database.mmsp-kn.de/kadid-10k-database.html), [LIVE challenge](https://live.ece.utexas.edu/research/ChallengeDB/), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html))



## Training

1. Pre-train model for EM. 

   ```python
   $ python train_pre.py
   ```

2. Final model for score prediction.

   ``` python
   $ python train_final.py
   ```




## Pretrained Models

Pretrained models will be released soon.



## Visualization

### 1. Predicted Error Maps

![supp1](./images/supp1.png)

![supp2](./images/supp2.png)

![supp3](./images/supp3.png)

![supp4](./images/supp4.png)



### 2. Perceptual Attention Maps

![supp5](./images/supp5.png)

![supp6](./images/supp6.png)

![supp7](./images/supp7.png)

![supp8](./images/supp8.png)
