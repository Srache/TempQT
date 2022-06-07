# Blind Image Quality Assessment via Transformer Predicted Error Map and Perceptual Quality Token

 ![Archi](./images/Archi.png)





## Environment

python3.8 + Pytorch1.11.0

> pip install -r  requirements



## Datasets

In this work we use 6 datasets ([LIVE](https://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23), [TID2013](http://www.ponomarenko.info/tid2013.htm), [KADID10K](http://database.mmsp-kn.de/kadid-10k-database.html), [LIVE challenge](https://live.ece.utexas.edu/research/ChallengeDB/), [KonIQ](http://database.mmsp-kn.de/koniq-10k-database.html), [LIVEFB](https://baidut.github.io/PaQ-2-PiQ/))



## Training

1. Pre-train model for EM. 

   ```python
   python train_pre.py
   ```

2. Final model for score prediction.

   ``` python
   python train_final.py
   ```




## Pretrained Models

Pretrained models will be released soon.



## Visualization

### 1. Predicted Error Maps

![supp1](C:\Users\Admin\Desktop\images\supp1.png)

![supp2](C:\Users\Admin\Desktop\images\supp2.png)

![supp3](C:\Users\Admin\Desktop\images\supp3.png)

![supp4](C:\Users\Admin\Desktop\images\supp4.png)



### 2. Perceptual Attention Maps

![supp5](C:\Users\Admin\Desktop\images\supp5.png)

![supp6](C:\Users\Admin\Desktop\images\supp6.png)

![supp7](C:\Users\Admin\Desktop\images\supp7.png)

![supp8](C:\Users\Admin\Desktop\images\supp8.png)
