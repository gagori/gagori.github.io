---
layout: post
title:  "[project3-8] OCR_Dataset "

---

# 한글 TEXT DATASET


**[ 데이터셋 ]**
OCR 대회를 주최하는 ICDAR 학회에서 제공하는 데이터셋 중 우리가 관심있는 Text Recognition Task의 데이터셋은 대다수가 영어로 구성되어 있다.   
아래는 2015년에 열린 ICDAR IC(Incidental Scene Text) 대회의 Task 4.3 Word Recognition 데이터셋이다.


![image](https://user-images.githubusercontent.com/86705085/146634671-39a223f3-e27a-4d05-a02e-54ab139d826e.png)


이와 같이 영어로 학습된 모델을 가져오면 한글인식은 당연히 떨어질 수 밖에 없다.
따라서 한글데이터셋을 따로 확보하여 fine-tuning 하는 과정이 필수적이다.
다행스럽게도 AI HUB에서 한글 이미지 데이터를 제공한다.

[한글 데이터셋 ] : https://aihub.or.kr/aidata/133

AI Hub에서 제공하는 한글 이미지 데이터는 **손글씨**, **인쇄체**, **Text in the Wild**로 세가지 종류가 있다. 
우리 프로젝트에서는 민증,여권 등 인쇄체를 인식하므로 인쇄체 이미지를 이용하여 학습할 것이다.




