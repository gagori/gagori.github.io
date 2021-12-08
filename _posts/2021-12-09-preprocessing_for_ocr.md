---
layout: post
title:  "[project3] preprocessing test"
---

# Adaptive Threshold & Gaussian Blur


Global Threshold는 문턱 값을 하나의 이미지 전체에 적용시키는 반면
Adaptive Threshold는 이미지의 구역구역마다 threshold를 실행 시켜줌.
  - https://m.blog.naver.com/samsjang/220504782549

추가 Gaussian Filter
  - https://m.blog.naver.com/samsjang/220505080672

### Global vs Adaptive TH 


```python
import cv2 

img_file='/content/drive/MyDrive/project_03/data/id.jpg'
img = cv2.imread(img_file, 0) 
ret, global_th = cv2.threshold(src=img, thresh=127, maxval=255, type=cv2.THRESH_BINARY) # 이미지 전체에 global 하게 적용되는 th=127
# print(ret)
thr1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # cv2.ADAPTIVE_THRESH_GAUSSIAN_C가 픽셀마다 th 찾아줌.
thr2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) 



g1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
g2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
g3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

g1_1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
g1_2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)
g1_3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 6)


cv2_imshow(g1)
cv2_imshow(g2)
cv2_imshow(g3)  # 선이 진해짐
cv2_imshow(g1_1)
cv2_imshow(g1_2)
cv2_imshow(g1_3)  # 흐려짐



# print("original gray scale image")
# cv2_imshow(img)
# print("global threshold")
# cv2_imshow(global_th) 
# print("gaussian threshold")
# cv2_imshow(thr1) # adaptive gaussian으로 이진화 한 것이 선을 더 잘 구분함을 확인.
# print("mean threshold")
# cv2_imshow(thr2) 
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_1.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_2.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_3.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_4.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_10_5.png)


### Gaussain Blur

Gaussian blur를 적용하면 히스토그램에서 확실한 봉우리를 만들고, 
여기에 Otsu 알고리즘을 적용하여 문턱값을 구한 후 
thresholding을 적용하면 보다 나은 Denoising이 된다.



```python
import matplotlib.pyplot as plt

def thresholding(img_file):
  img = cv2.imread(img_file, 0) # gray scale

  # global threshold
  ret, thr1 = cv2.threshold(img, 127,255,cv2.THRESH_BINARY)
  # Otsu binarization
  ret, thr2 = cv2.threshold(img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  # Gaussian blur + Otsu
  blur = cv2.GaussianBlur(img, (3,3), 0)  # kernel 커질수록 노이즈는 사라지지만 그만큼 정보손실 큼. (kernel size : 홀수)
  ret, thr3 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  titles = ['original noisy','Histogram','Global Thresholding',
            'original noisy', 'Histogram', 'Otsu Thresholding',
            'Gaussian-filtered','Histogram', 'Otsu Thresholding']

  images = [img, 0, thr1, img, 0, thr2, blur, 0, thr3]

  plt.figure(figsize=(15,10))
  for i in range(3):
    plt.subplot(3,3, i*3+1), plt.imshow(images[i*3], 'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3, i*3+2), plt.hist(images[i*3].ravel(), 256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])

    plt.subplot(3,3, i*3+3), plt.imshow(images[i*3+2], 'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
  
  plt.show()

thresholding(img_file)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_12_0.png)



```python
'''
Gaussian blur를 적용하면 히스토그램에서 확실한 봉우리를 만들고, 
여기에 Otsu 알고리즘을 적용하여 문턱값을 구한 후 
thresholding을 적용하면 보다 나은 Denoising이 된다.
'''
```

### Gaussain blur + Otsu binaryzation


```python
def image_smoothening(img,_ThredholdCount=0):
  ret1, th1 = cv2.threshold(img, G_BINARY_THREHOLD+(G_BINARY_THREHOLD_ALPHA*_ThredholdCount), 255, cv2.THRESH_BINARY)
  ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  blur = cv2.GaussianBlur(th2, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
  ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  titles = ['Global','Otsu','Otsu > Gaussain_Blur','Otsu > Gaussian_Blur > Otsu']
  images = [th1,th2,blur,th3]
  plt.figure(figsize=(15,10))
  for i in range(4):
    plt.subplot(2,2, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])

  plt.show()


img = cv2.imread(img_file, 0) # gray scale 불러와야함.
image_smoothening(img)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_15_0.png)


# 전처리 후 Bounding Box


```python
# 이미지 전처리
'''
1. Gaussian Filter
2. Otsu Adaptive Threshold
'''
img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file, 0)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# cv2_imshow(img)

# original_img = cv2.imread(img_file)
# gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
# cv2_imshow(blur)
# ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의
# cv2_imshow(th1)

blur = cv2.GaussianBlur(img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
cv2_imshow(blur)
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의
cv2_imshow(th1)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_17_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_17_1.png)



```python
import cv2
import pytesseract
from google.colab.patches import cv2_imshow

img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file)
# img = cv2.resize(img, dsize=None, fx=1.25, fy=1.25)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의


config = r'--oem 3 --psm 6 outputbase digits' 
# boxes = pytesseract.pytesseract.image_to_string(th1, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(th1, lang='kor+eng', config=config)  # 숫자만
# print(boxes)
# print(boxes_num)


for idx, b in enumerate(boxes_num.splitlines()):
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        # print(b)
        if len(b) ==12:  # 객체가 있는것만 뽑아서
            print(b)
            # if ('-' in b[11]) and ('.' not in b[11]) :
            # if ('-' in b[11]) and (len(b[11]) >13):
            if '-' in b[11] :
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                # print(x,y,w,h)
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)

# boxes_list = []
# for b in boxes.splitlines():
#     '''
#     인식된 text를 모두 리스트에 담아, 
#     필요한 정보만 indexing 할 수 있는 알고리즘 
#     '''
#     # print(b)
#     boxes_list.append(b)  # list에 담으면 indexing으로 원하는 것만 가져오기 편하므로
# print(boxes_list)  # 만약 깨진거 있으니까 frozen_east_text_detection 딥러닝으로 pretrained된 모델을 가져와서 쓰면 잘 인식함.


cv2_imshow(img)
```

    ['5', '1', '1', '1', '1', '1', '20', '35', '6', '16', '93', '1']
    ['5', '1', '1', '1', '1', '2', '30', '34', '20', '18', '84', '종']
    ['5', '1', '1', '1', '1', '3', '54', '33', '41', '19', '93', '대형']
    ['5', '1', '1', '1', '1', '4', '0', '0', '900', '623', '93', '1']
    ['5', '1', '1', '1', '1', '5', '113', '34', '29', '18', '93', '종']
    ['5', '1', '1', '1', '1', '6', '146', '34', '42', '18', '66', '보통']
    ['5', '1', '1', '1', '1', '7', '0', '0', '900', '623', '89', '1']
    ['5', '1', '1', '1', '1', '8', '204', '34', '30', '18', '93', '종']
    ['5', '1', '1', '1', '1', '9', '238', '33', '41', '19', '96', '소형']
    ['5', '1', '1', '1', '2', '1', '19', '75', '20', '19', '82', '특']
    ['5', '1', '1', '1', '2', '2', '42', '74', '20', '20', '93', '수']
    ['5', '1', '1', '1', '2', '3', '0', '0', '900', '623', '93', '(']
    ['5', '1', '1', '1', '2', '4', '66', '73', '29', '22', '93', '대']
    ['5', '1', '1', '1', '2', '5', '100', '74', '18', '20', '92', '형']
    ['5', '1', '1', '1', '2', '6', '123', '74', '18', '20', '92', '견']
    ['5', '1', '1', '1', '2', '7', '147', '74', '17', '20', '90', '인']
    ['5', '1', '1', '1', '2', '8', '0', '0', '900', '623', '91', ',']
    ['5', '1', '1', '1', '2', '9', '171', '75', '29', '20', '92', '소']
    ['5', '1', '1', '1', '2', '10', '204', '74', '18', '20', '91', '형']
    ['5', '1', '1', '1', '2', '11', '227', '74', '17', '20', '91', '견']
    ['5', '1', '1', '1', '2', '12', '250', '74', '18', '20', '93', '인']
    ['5', '1', '1', '1', '2', '13', '0', '0', '900', '623', '81', ',']
    ['5', '1', '1', '1', '2', '14', '276', '75', '28', '20', '93', '구']
    ['5', '1', '1', '1', '2', '15', '308', '74', '19', '20', '93', '난']
    ['5', '1', '1', '1', '2', '16', '332', '73', '6', '22', '93', ')']
    ['5', '1', '1', '1', '2', '17', '414', '63', '27', '29', '93', '자']
    ['5', '1', '1', '1', '2', '18', '443', '63', '25', '30', '91', '동']
    ['5', '1', '1', '1', '2', '19', '473', '61', '27', '31', '93', '차']
    ['5', '1', '1', '1', '2', '20', '502', '61', '25', '31', '93', '운']
    ['5', '1', '1', '1', '2', '21', '532', '63', '24', '29', '93', '전']
    ['5', '1', '1', '1', '2', '22', '562', '61', '83', '32', '11', '면허증']
    ['5', '1', '1', '1', '2', '23', '653', '69', '90', '23', '0', '(00606']
    ['5', '1', '1', '1', '2', '24', '750', '69', '89', '23', '65', '1106056)']
    ['5', '1', '1', '1', '3', '1', '17', '116', '79', '20', '0', '2oas']
    ['5', '1', '1', '1', '3', '2', '110', '116', '76', '19', '68', '2648']
    ['5', '1', '1', '1', '3', '3', '203', '116', '65', '19', '59', '85']
    ['5', '1', '1', '1', '3', '4', '356', '114', '424', '38', '59', '21-19-174133-01']
    ['5', '1', '1', '1', '4', '1', '351', '175', '30', '30', '92', '홍']
    ['5', '1', '1', '1', '4', '2', '388', '175', '28', '30', '92', '길']
    ['5', '1', '1', '1', '4', '3', '425', '175', '31', '30', '93', '순']
    ['5', '1', '1', '1', '5', '1', '2', '210', '12', '43', '75', ':']
    ['5', '1', '1', '1', '5', '2', '349', '210', '140', '56', '48', '000829"']
    ['5', '1', '1', '1', '5', '3', '490', '215', '142', '48', '48', '2134567']
    ['5', '1', '1', '1', '6', '1', '2', '264', '14', '37', '30', '<']
    ['5', '1', '1', '1', '6', '2', '350', '258', '98', '45', '55', 'MBA']
    ['5', '1', '1', '1', '6', '3', '457', '255', '135', '47', '47', 'AUS']
    ['5', '1', '1', '1', '6', '4', '586', '265', '113', '37', '59', 'Sas']
    ['5', '1', '1', '1', '7', '1', '2', '306', '28', '32', '18', 'MS']
    ['5', '1', '1', '1', '7', '2', '341', '301', '45', '41', '21', 'S72']
    ['5', '1', '1', '1', '7', '3', '383', '300', '124', '35', '21', '(02S']
    ['5', '1', '1', '1', '7', '4', '508', '300', '100', '41', '65', '29,']
    ['5', '1', '1', '1', '7', '5', '703', '316', '1', '4', '51', 'ㆍ']
    ['5', '1', '1', '1', '7', '6', '0', '0', '900', '623', '34', '/']
    ['5', '1', '1', '1', '7', '7', '769', '290', '86', '70', '34', '^']
    ['5', '1', '1', '1', '8', '1', '2', '363', '26', '19', '41', '내']
    ['5', '1', '1', '1', '8', '2', '122', '362', '123', '13', '21', '스즈']
    ['5', '1', '1', '1', '8', '3', '336', '364', '62', '22', '30', 'ee']
    ['5', '1', '1', '1', '8', '4', '528', '365', '84', '23', '23', 'Bed']
    ['5', '1', '1', '1', '9', '1', '2', '385', '23', '36', '20', '=']
    ['5', '1', '1', '1', '9', '2', '168', '418', '27', '3', '14', '..']
    ['5', '1', '1', '1', '9', '3', '343', '385', '128', '41', '0', 'BSAA']
    ['5', '1', '1', '1', '9', '4', '516', '389', '174', '37', '32', '2029-01.01.']
    ['5', '1', '1', '1', '9', '5', '716', '396', '34', '12', '74', '—']
    ['5', '1', '1', '1', '10', '1', '2', '466', '30', '34', '92', '=']
    ['5', '1', '1', '1', '10', '2', '347', '458', '30', '34', '90', '조']
    ['5', '1', '1', '1', '10', '3', '378', '458', '38', '39', '63', '“']
    ['5', '1', '1', '1', '10', '4', '443', '463', '60', '38', '55', 'War']
    ['5', '1', '1', '1', '10', '5', '513', '460', '65', '40', '16', 'fe']
    ['5', '1', '1', '1', '10', '6', '753', '456', '125', '29', '72', '8H1X3Y']
    ['5', '1', '1', '1', '11', '1', '2', '562', '129', '55', '18', '결정기']
    ['5', '1', '1', '1', '11', '2', '151', '595', '4', '5', '31', ':']
    ['5', '1', '1', '1', '11', '3', '172', '587', '23', '21', '86', '조']
    ['5', '1', '1', '1', '11', '4', '201', '587', '42', '22', '89', '직']
    ['5', '1', '1', '1', '11', '5', '247', '587', '34', '23', '43', '기']
    ['5', '1', '1', '1', '11', '6', '329', '574', '138', '29', '80', '2019.09.']
    ['5', '1', '1', '1', '11', '7', '479', '564', '264', '38', '15', '0oNSNSS']
    ['5', '1', '1', '1', '11', '8', '756', '562', '128', '40', '29', 'Shee']



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_18_1.png)


### Resize 이후 인식 잘됨을 확인


```python
import cv2
import pytesseract
from google.colab.patches import cv2_imshow


img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file)
img = cv2.resize(img, dsize=None, fx=1.25, fy=1.25)  # Resize
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
ret1, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # gray scale을 받음에 주의


config = r'--oem 3 --psm 6 outputbase digits'
boxes = pytesseract.pytesseract.image_to_string(th1, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(th1, lang='kor+eng', config=config)  # 숫자만

num_list=[]
for idx, b in enumerate(boxes_num.splitlines()): # 한 줄 씩 split
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        # print(b)
        if len(b) == 12:  # 객체가 있는것만 뽑아서
            if ('.' not in b[11]) and (len(b[11])>=13):
            # if ('-' in b[11]) :
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)



cv2_imshow(img)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_20_0.png)


# 다양한 전처리 시도
이후에서는 인식률을 높이기 위한 다양한 방법을 연습해보겠습니다. 
  - Denoising
    - Gaussian filtering
    - Sharpening
    - Adaptive thresholding
  - Edge detection
    - Sobel
    - Canny


### gaussian filtering & sharpening


```python
# gaussian filtering & sharpening
'''
1. Gaussian Filter
2. Sharpening
'''
img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file, 0)
blur = cv2.GaussianBlur(img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
sharp = np.clip(2.0*img - blur, 0, 255).astype(np.uint8)

images = [img, blur, sharp]
titles= ['original','gaussian blur','sharpening']
plt.figure(figsize=(15,30))
for i in range(3):
  plt.subplot(3,1,i+1), plt.imshow(images[i], 'gray')
  plt.title(titles[i]), plt.xticks([]), plt.yticks([])
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_23_0.png)



```python
# binarization

ret1, th1 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(sharp, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# images=[th1, th2]
# titles=["blur > otsu", "blur > sharpening > otsu"]
# plt.figure(figsize=(30,15))
# for i in range(2):
#   plt.subplot(1,2,i+1), plt.imshow(images[i], "gray")
#   plt.title(titles[i]), plt.xticks([]), plt.yticks([])

print("blur > otsu")
cv2_imshow(th1) 
print("blur > sharpening > otsu")
cv2_imshow(th2)   # 기존 blur보다 진해졌음.
```

    blur > otsu



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_24_1.png)


    blur > sharpening > otsu



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_24_3.png)



```python
import cv2
import pytesseract
from google.colab.patches import cv2_imshow

img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
sharp = np.clip(2.0*gray_img - blur, 0, 255).astype(np.uint8)
ret1, th1 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(sharp, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


########################### text detection algorithm adaption ######################################
config = r'--oem 3 --psm 6 outputbase digits' # 숫자만 
# boxes = pytesseract.pytesseract.image_to_string(th1, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(th2, lang='kor+eng', config=config)  # 숫자만
# print(boxes)
# print(boxes_num)

# num_list=[]
for idx, b in enumerate(boxes_num.splitlines()): # 한 줄 씩 split
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        if len(b) == 12:  # 객체가 있는것만 뽑아서
            print(b)
            # if ('.' not in b[11]) and (len(b[11])>=13):
            # if ('-' in b[11]) :
            if len(b[11]) >=6:
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)



cv2_imshow(img)
```

    ['5', '1', '1', '1', '1', '1', '20', '35', '6', '16', '93', '1']
    ['5', '1', '1', '1', '1', '2', '30', '34', '20', '18', '84', '종']
    ['5', '1', '1', '1', '1', '3', '54', '33', '41', '19', '93', '대형']
    ['5', '1', '1', '1', '1', '4', '0', '0', '900', '623', '93', '1']
    ['5', '1', '1', '1', '1', '5', '113', '34', '29', '18', '93', '종']
    ['5', '1', '1', '1', '1', '6', '146', '34', '42', '18', '66', '보통']
    ['5', '1', '1', '1', '1', '7', '0', '0', '900', '623', '89', '1']
    ['5', '1', '1', '1', '1', '8', '204', '34', '30', '18', '93', '종']
    ['5', '1', '1', '1', '1', '9', '238', '33', '41', '19', '96', '소형']
    ['5', '1', '1', '1', '2', '1', '19', '75', '20', '19', '82', '특']
    ['5', '1', '1', '1', '2', '2', '42', '74', '20', '20', '93', '수']
    ['5', '1', '1', '1', '2', '3', '0', '0', '900', '623', '93', '(']
    ['5', '1', '1', '1', '2', '4', '66', '73', '29', '22', '93', '대']
    ['5', '1', '1', '1', '2', '5', '100', '74', '18', '20', '92', '형']
    ['5', '1', '1', '1', '2', '6', '123', '74', '18', '20', '92', '견']
    ['5', '1', '1', '1', '2', '7', '147', '74', '17', '20', '90', '인']
    ['5', '1', '1', '1', '2', '8', '0', '0', '900', '623', '91', ',']
    ['5', '1', '1', '1', '2', '9', '171', '75', '29', '20', '92', '소']
    ['5', '1', '1', '1', '2', '10', '204', '74', '18', '20', '91', '형']
    ['5', '1', '1', '1', '2', '11', '227', '74', '17', '20', '91', '견']
    ['5', '1', '1', '1', '2', '12', '250', '74', '18', '20', '93', '인']
    ['5', '1', '1', '1', '2', '13', '0', '0', '900', '623', '81', ',']
    ['5', '1', '1', '1', '2', '14', '276', '75', '28', '20', '93', '구']
    ['5', '1', '1', '1', '2', '15', '308', '74', '19', '20', '93', '난']
    ['5', '1', '1', '1', '2', '16', '332', '73', '6', '22', '93', ')']
    ['5', '1', '1', '1', '2', '17', '414', '63', '27', '29', '93', '자']
    ['5', '1', '1', '1', '2', '18', '443', '63', '25', '30', '91', '동']
    ['5', '1', '1', '1', '2', '19', '473', '61', '27', '31', '93', '차']
    ['5', '1', '1', '1', '2', '20', '502', '61', '25', '31', '93', '운']
    ['5', '1', '1', '1', '2', '21', '532', '63', '24', '29', '93', '전']
    ['5', '1', '1', '1', '2', '22', '562', '61', '83', '32', '11', '면허증']
    ['5', '1', '1', '1', '2', '23', '653', '69', '90', '23', '0', '(00606']
    ['5', '1', '1', '1', '2', '24', '750', '69', '89', '23', '65', '1106056)']
    ['5', '1', '1', '1', '3', '1', '17', '116', '79', '20', '0', '2oas']
    ['5', '1', '1', '1', '3', '2', '110', '116', '76', '19', '68', '2648']
    ['5', '1', '1', '1', '3', '3', '203', '116', '65', '19', '59', '85']
    ['5', '1', '1', '1', '3', '4', '356', '114', '424', '38', '59', '21-19-174133-01']
    ['5', '1', '1', '1', '4', '1', '351', '175', '30', '30', '92', '홍']
    ['5', '1', '1', '1', '4', '2', '388', '175', '28', '30', '92', '길']
    ['5', '1', '1', '1', '4', '3', '425', '175', '31', '30', '93', '순']
    ['5', '1', '1', '1', '5', '1', '2', '210', '12', '43', '75', ':']
    ['5', '1', '1', '1', '5', '2', '349', '210', '140', '56', '48', '000829"']
    ['5', '1', '1', '1', '5', '3', '490', '215', '142', '48', '48', '2134567']
    ['5', '1', '1', '1', '6', '1', '2', '264', '14', '37', '30', '<']
    ['5', '1', '1', '1', '6', '2', '350', '258', '98', '45', '55', 'MBA']
    ['5', '1', '1', '1', '6', '3', '457', '255', '135', '47', '47', 'AUS']
    ['5', '1', '1', '1', '6', '4', '586', '265', '113', '37', '59', 'Sas']
    ['5', '1', '1', '1', '7', '1', '2', '306', '28', '32', '18', 'MS']
    ['5', '1', '1', '1', '7', '2', '341', '301', '45', '41', '21', 'S72']
    ['5', '1', '1', '1', '7', '3', '383', '300', '124', '35', '21', '(02S']
    ['5', '1', '1', '1', '7', '4', '508', '300', '100', '41', '65', '29,']
    ['5', '1', '1', '1', '7', '5', '703', '316', '1', '4', '51', 'ㆍ']
    ['5', '1', '1', '1', '7', '6', '0', '0', '900', '623', '34', '/']
    ['5', '1', '1', '1', '7', '7', '769', '290', '86', '70', '34', '^']
    ['5', '1', '1', '1', '8', '1', '2', '363', '26', '19', '41', '내']
    ['5', '1', '1', '1', '8', '2', '122', '362', '123', '13', '21', '스즈']
    ['5', '1', '1', '1', '8', '3', '336', '364', '62', '22', '30', 'ee']
    ['5', '1', '1', '1', '8', '4', '528', '365', '84', '23', '23', 'Bed']
    ['5', '1', '1', '1', '9', '1', '2', '385', '23', '36', '20', '=']
    ['5', '1', '1', '1', '9', '2', '168', '418', '27', '3', '14', '..']
    ['5', '1', '1', '1', '9', '3', '343', '385', '128', '41', '0', 'BSAA']
    ['5', '1', '1', '1', '9', '4', '516', '389', '174', '37', '32', '2029-01.01.']
    ['5', '1', '1', '1', '9', '5', '716', '396', '34', '12', '74', '—']
    ['5', '1', '1', '1', '10', '1', '2', '466', '30', '34', '92', '=']
    ['5', '1', '1', '1', '10', '2', '347', '458', '30', '34', '90', '조']
    ['5', '1', '1', '1', '10', '3', '378', '458', '38', '39', '63', '“']
    ['5', '1', '1', '1', '10', '4', '443', '463', '60', '38', '55', 'War']
    ['5', '1', '1', '1', '10', '5', '513', '460', '65', '40', '16', 'fe']
    ['5', '1', '1', '1', '10', '6', '753', '456', '125', '29', '72', '8H1X3Y']
    ['5', '1', '1', '1', '11', '1', '2', '562', '129', '55', '18', '결정기']
    ['5', '1', '1', '1', '11', '2', '151', '595', '4', '5', '31', ':']
    ['5', '1', '1', '1', '11', '3', '172', '587', '23', '21', '86', '조']
    ['5', '1', '1', '1', '11', '4', '201', '587', '42', '22', '89', '직']
    ['5', '1', '1', '1', '11', '5', '247', '587', '34', '23', '43', '기']
    ['5', '1', '1', '1', '11', '6', '329', '574', '138', '29', '80', '2019.09.']
    ['5', '1', '1', '1', '11', '7', '479', '564', '264', '38', '15', '0oNSNSS']
    ['5', '1', '1', '1', '11', '8', '756', '562', '128', '40', '29', 'Shee']



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_25_1.png)


##### sharpening 별 효과 없었음 확인
  - 눈으로는 txt와 배경의 차이가 강해져서 더 잘잡을 것이라 생각함
  - 그러나 결과는 별 차이 없음 (거의 동일)
  - 눈으로 확인할때와 컴퓨터가 인식할때는 확실히 차이가 있나보다...

### Edge detection 적용
  - edge란 영상에서 픽셀값이 급격하게 변하는 부분
  - 일반적으로 배경과 객체, 또는 객체와 객체의 경계
  - 영상을 (x,y)함수로 간주했을 때, foc값이 크게 나타나는 부분을 검출


#### Sobel


```python
# Sobel

dx = cv2.Sobel(th1, -1, 1,0, delta=0)  # x방향
dy = cv2.Sobel(th1, -1, 0,1, delta=255)  # y방향

cv2_imshow(dx)
cv2_imshow(dy)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_29_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_29_1.png)



```python
dx = cv2.Sobel(th1, cv2.CV_32F, 1,0)  # x방향. delta default = 0
dy = cv2.Sobel(th1, cv2.CV_32F, 0,1)  # y방향

# 방향상관없이 gradiant 크기 보기
mag = cv2.magnitude(dx, dy)
mag = np.clip(mag, 0, 255).astype(np.uint8)

# _, th3 = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 이미 이진화 되서 별 의미 없음
# _, th3 = cv2.threshold(mag, 120, 255, cv2.THRESH_BINARY)

cv2_imshow(th1)
cv2_imshow(mag)
# cv2_imshow(th3)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_30_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_30_1.png)



```python

```

#### Canny
(순서)
  - 가우시안 필터링 > 그래디언트 계산 > NMS >이중 임계값을 이용한 히스테리시스 에지 트래킹



```python
# Canny
canny = cv2.Canny(th1, 50,150)

cv2_imshow(th1)
cv2_imshow(canny)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_33_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_33_1.png)



```python
import cv2
import pytesseract
from google.colab.patches import cv2_imshow

img_file='/content/drive/MyDrive/project_03/data/driver3.jpg'
img = cv2.imread(img_file)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_img, (1, 1), 0)  # kernel size 크게 잡지 않도록 주의
sharp = np.clip(2.0*gray_img - blur, 0, 255).astype(np.uint8)
ret1, th1 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret2, th2 = cv2.threshold(sharp, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
dx = cv2.Sobel(th1, cv2.CV_32F, 1,0)  # x방향. delta default = 0
dy = cv2.Sobel(th1, cv2.CV_32F, 0,1)  # y방향
mag = cv2.magnitude(dx, dy) # 방향상관없이
mag = np.clip(mag, 0, 255).astype(np.uint8)
canny = cv2.Canny(th1, 50, 150)
mp1 = cv2.dilate(th2, None)
se= cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
mp2 = cv2.erode(th1, se)


########################### text detection algorithm adaption ######################################
config = r'--oem 3 --psm 6 outputbase digits' # 숫자만 
# boxes = pytesseract.pytesseract.image_to_string(th1, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(mp2, lang='kor+eng', config=config)  # 숫자만
# print(boxes)
# print(boxes_num)

# num_list=[]
for idx, b in enumerate(boxes_num.splitlines()): # 한 줄 씩 split
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        if len(b) == 12:  # 객체가 있는것만 뽑아서
            print(b)
            # if ('.' not in b[11]) and (len(b[11])>=13):
            # if ('-' in b[11]):
            if (len(b[11]) >= 6) and ('.' not in b[11]):
                # print(b)
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단 >> 시작점, 끝점이니까 좌하단 우상단 해도 상관 없음.
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)


cv2_imshow(mp2)
cv2_imshow(img)
```

    ['5', '1', '1', '1', '1', '1', '20', '35', '6', '16', '93', '1']
    ['5', '1', '1', '1', '1', '2', '30', '34', '20', '18', '84', '종']
    ['5', '1', '1', '1', '1', '3', '54', '33', '41', '19', '93', '대형']
    ['5', '1', '1', '1', '1', '4', '0', '0', '900', '623', '93', '1']
    ['5', '1', '1', '1', '1', '5', '113', '34', '29', '18', '93', '종']
    ['5', '1', '1', '1', '1', '6', '146', '34', '42', '18', '66', '보통']
    ['5', '1', '1', '1', '1', '7', '0', '0', '900', '623', '89', '1']
    ['5', '1', '1', '1', '1', '8', '204', '34', '30', '18', '93', '종']
    ['5', '1', '1', '1', '1', '9', '238', '33', '41', '19', '96', '소형']
    ['5', '1', '1', '1', '2', '1', '19', '75', '20', '19', '82', '특']
    ['5', '1', '1', '1', '2', '2', '42', '74', '20', '20', '93', '수']
    ['5', '1', '1', '1', '2', '3', '0', '0', '900', '623', '93', '(']
    ['5', '1', '1', '1', '2', '4', '66', '73', '29', '22', '93', '대']
    ['5', '1', '1', '1', '2', '5', '100', '74', '18', '20', '92', '형']
    ['5', '1', '1', '1', '2', '6', '123', '74', '18', '20', '92', '견']
    ['5', '1', '1', '1', '2', '7', '147', '74', '17', '20', '90', '인']
    ['5', '1', '1', '1', '2', '8', '0', '0', '900', '623', '91', ',']
    ['5', '1', '1', '1', '2', '9', '171', '75', '29', '20', '92', '소']
    ['5', '1', '1', '1', '2', '10', '204', '74', '18', '20', '91', '형']
    ['5', '1', '1', '1', '2', '11', '227', '74', '17', '20', '91', '견']
    ['5', '1', '1', '1', '2', '12', '250', '74', '18', '20', '93', '인']
    ['5', '1', '1', '1', '2', '13', '0', '0', '900', '623', '81', ',']
    ['5', '1', '1', '1', '2', '14', '276', '75', '28', '20', '93', '구']
    ['5', '1', '1', '1', '2', '15', '308', '74', '19', '20', '93', '난']
    ['5', '1', '1', '1', '2', '16', '332', '73', '6', '22', '93', ')']
    ['5', '1', '1', '1', '2', '17', '414', '63', '27', '29', '93', '자']
    ['5', '1', '1', '1', '2', '18', '443', '63', '25', '30', '91', '동']
    ['5', '1', '1', '1', '2', '19', '473', '61', '27', '31', '93', '차']
    ['5', '1', '1', '1', '2', '20', '502', '61', '25', '31', '93', '운']
    ['5', '1', '1', '1', '2', '21', '532', '63', '24', '29', '93', '전']
    ['5', '1', '1', '1', '2', '22', '562', '61', '83', '32', '11', '면허증']
    ['5', '1', '1', '1', '2', '23', '653', '69', '90', '23', '0', '(00606']
    ['5', '1', '1', '1', '2', '24', '750', '69', '89', '23', '65', '1106056)']
    ['5', '1', '1', '1', '3', '1', '17', '116', '79', '20', '0', '2oas']
    ['5', '1', '1', '1', '3', '2', '110', '116', '76', '19', '68', '2648']
    ['5', '1', '1', '1', '3', '3', '203', '116', '65', '19', '59', '85']
    ['5', '1', '1', '1', '3', '4', '356', '114', '424', '38', '59', '21-19-174133-01']
    ['5', '1', '1', '1', '4', '1', '351', '175', '30', '30', '92', '홍']
    ['5', '1', '1', '1', '4', '2', '388', '175', '28', '30', '92', '길']
    ['5', '1', '1', '1', '4', '3', '425', '175', '31', '30', '93', '순']
    ['5', '1', '1', '1', '5', '1', '2', '210', '12', '43', '75', ':']
    ['5', '1', '1', '1', '5', '2', '349', '210', '140', '56', '48', '000829"']
    ['5', '1', '1', '1', '5', '3', '490', '215', '142', '48', '48', '2134567']
    ['5', '1', '1', '1', '6', '1', '2', '264', '14', '37', '30', '<']
    ['5', '1', '1', '1', '6', '2', '350', '258', '98', '45', '55', 'MBA']
    ['5', '1', '1', '1', '6', '3', '457', '255', '135', '47', '47', 'AUS']
    ['5', '1', '1', '1', '6', '4', '586', '265', '113', '37', '59', 'Sas']
    ['5', '1', '1', '1', '7', '1', '2', '306', '28', '32', '18', 'MS']
    ['5', '1', '1', '1', '7', '2', '341', '301', '45', '41', '21', 'S72']
    ['5', '1', '1', '1', '7', '3', '383', '300', '124', '35', '21', '(02S']
    ['5', '1', '1', '1', '7', '4', '508', '300', '100', '41', '65', '29,']
    ['5', '1', '1', '1', '7', '5', '703', '316', '1', '4', '51', 'ㆍ']
    ['5', '1', '1', '1', '7', '6', '0', '0', '900', '623', '34', '/']
    ['5', '1', '1', '1', '7', '7', '769', '290', '86', '70', '34', '^']
    ['5', '1', '1', '1', '8', '1', '2', '363', '26', '19', '41', '내']
    ['5', '1', '1', '1', '8', '2', '122', '362', '123', '13', '21', '스즈']
    ['5', '1', '1', '1', '8', '3', '336', '364', '62', '22', '30', 'ee']
    ['5', '1', '1', '1', '8', '4', '528', '365', '84', '23', '23', 'Bed']
    ['5', '1', '1', '1', '9', '1', '2', '385', '23', '36', '20', '=']
    ['5', '1', '1', '1', '9', '2', '168', '418', '27', '3', '14', '..']
    ['5', '1', '1', '1', '9', '3', '343', '385', '128', '41', '0', 'BSAA']
    ['5', '1', '1', '1', '9', '4', '516', '389', '174', '37', '32', '2029-01.01.']
    ['5', '1', '1', '1', '9', '5', '716', '396', '34', '12', '74', '—']
    ['5', '1', '1', '1', '10', '1', '2', '466', '30', '34', '92', '=']
    ['5', '1', '1', '1', '10', '2', '347', '458', '30', '34', '90', '조']
    ['5', '1', '1', '1', '10', '3', '378', '458', '38', '39', '63', '“']
    ['5', '1', '1', '1', '10', '4', '443', '463', '60', '38', '55', 'War']
    ['5', '1', '1', '1', '10', '5', '513', '460', '65', '40', '16', 'fe']
    ['5', '1', '1', '1', '10', '6', '753', '456', '125', '29', '72', '8H1X3Y']
    ['5', '1', '1', '1', '11', '1', '2', '562', '129', '55', '18', '결정기']
    ['5', '1', '1', '1', '11', '2', '151', '595', '4', '5', '31', ':']
    ['5', '1', '1', '1', '11', '3', '172', '587', '23', '21', '86', '조']
    ['5', '1', '1', '1', '11', '4', '201', '587', '42', '22', '89', '직']
    ['5', '1', '1', '1', '11', '5', '247', '587', '34', '23', '43', '기']
    ['5', '1', '1', '1', '11', '6', '329', '574', '138', '29', '80', '2019.09.']
    ['5', '1', '1', '1', '11', '7', '479', '564', '264', '38', '15', '0oNSNSS']
    ['5', '1', '1', '1', '11', '8', '756', '562', '128', '40', '29', 'Shee']



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_34_1.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_34_2.png)



```python

```

#### Morphology


```python
# morphology

mp1 = cv2.dilate(th1, None)
se= cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
mp2 = cv2.erode(th1, se)

cv2_imshow(mp1)
cv2_imshow(mp2)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_37_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_37_1.png)



```python
se= cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
mp2 = cv2.erode(th1, se)


cv2_imshow(th1)
cv2_imshow(mp2)
```


![png](preprocessing_for_ocr_files/preprocessing_for_ocr_38_0.png)



![png](preprocessing_for_ocr_files/preprocessing_for_ocr_38_1.png)



```python
%%shell
jupyter nbconvert --to markdown /content/text_detection_tesseract.ipynb
```
