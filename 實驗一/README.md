# 實驗一

[TOC]

## Enviroment

- OS : ubuntu 18.04 LTS
- GPU: GTX 1070 8G
- RAM: 16G

## Data set

- train dataset : 764
- test dataset: 32
- word embedding file: 
    - word size: 1292608
    - [Reference](https://github.com/Embedding/Chinese-Word-Vectors)
    - Mixed-large 综合

## Method

- 將 training data 各個資料尾巴加隨機詞做擴充
- label 二進位轉成十進位方式 (ex:  101 -> 5)，multilabel 轉成 一個 label


## Result

![](https://i.imgur.com/N0UVoAH.png)


## 結論

- 正向情緒部份不太準確
- 負面情緒尚可
