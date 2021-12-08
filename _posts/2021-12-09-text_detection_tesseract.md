---
layout: post
title:  "[project3] idcard deidentification"
---

# tesseract를 통해 이미지를 식별하고 bbox를 사용하여 주요정보를 가리는 것을 연습합니다.

# tesseract 설치
  - https://velog.io/@latte_h/Tesseract


```python
from google.colab import drive
drive.mount("/content/drive")
```


```python
!sudo apt install tesseract-ocr
```


```python
!pip install pillow
```


```python
!pip install pytesseract
```

# 필요한 라이브러리


```python
from PIL import Image
import pytesseract
import cv2
from google.colab.patches import cv2_imshow
# 코랩에서 cv2.imshow 지원하지 않음
```

# pytesseract 연습
  - image_to_string
  - image_to_data
  - image_to_boxes


```python
img = Image.open('/content/drive/MyDrive/project_03/data/id.jpg')        
data = pytesseract.pytesseract.image_to_string(img, lang="kor+eng")   

print(data)
# 숫자는 잘 인식하지만 한글은 잘 인식 못함
# 아래서 한글깨짐 방지 적용

# with open("sample.txt", "w") as f:
#     f.write(text)
```

    FUSES
    SAS (KAM)
    
    800101-2345678
    
    Alesana eS
    (Aguaee is)
    
     
    
    2020.08.16.
    AGS SATAY
    


#### 한글깨짐 방지
  - https://webnautes.tistory.com/947


```python
!sudo apt install tesseract-ocr-kor
```


```python
!sudo apt install tesseract-ocr-script-hang
```


```python
# 방법1
data = pytesseract.pytesseract.image_to_string(img, lang="Hangul")
print(data)
```

    주민등록증
    홍 길 동 ( 빠 솜 0)
    
    800101-2345678
    
    서 울 특별시 가 산 디 지 털 1 로
    ( 대 륭 데 크 노 타운 18 차 )
    
     
    
    2020.08.16.
    서 올 특별시 금 천 구 청 장 |
    



```python
# 방법2
config = r'--oem 3 --psm 6 outputbase digits' # 숫자인식 옵션
data = pytesseract.pytesseract.image_to_string(img, lang="kor+eng", config = config)   

print(data)
```

    주민등록증
    홍길동(래좀0)
    800101-2345678
    서울특별시 가산디지털1로
    (ASHLEE 18%)         fe.
    2020.08.16.   Aa
    서울특별시 로!
    



```python
# image_to_boxes
# 식별된 text와 좌표정보가 같이 출력됨
data = pytesseract.pytesseract.image_to_boxes(img, lang="kor+eng", config = config)   
print(data)
```

    주 42 145 58 161 0
    민 63 145 78 161 0
    등 84 145 100 161 0
    록 104 145 120 161 0
    증 130 145 141 160 0
    홍 29 118 43 132 0
    길 45 118 58 132 0
    동 61 118 73 131 0
    ( 77 119 80 131 0
    래 85 119 98 131 0
    좀 101 119 114 132 0
    0 116 118 126 131 0
    ) 133 118 137 132 0
    8 28 98 33 108 0
    0 36 98 40 108 0
    0 44 98 49 108 0
    1 53 98 57 108 0
    0 60 98 64 108 0
    1 70 98 72 108 0
    - 77 102 79 104 0
    2 82 98 86 108 0
    3 90 98 94 108 0
    4 98 98 103 108 0
    5 107 98 110 108 0
    6 110 98 112 108 0
    7 117 98 121 108 0
    8 125 98 137 108 0
    서 22 78 30 88 0
    울 33 78 41 88 0
    특 43 78 52 88 0
    별 54 78 62 88 0
    시 65 78 71 88 0
    가 80 80 85 86 0
    산 85 78 98 88 0
    디 101 78 108 88 0
    지 111 78 119 88 0
    털 121 78 127 88 0
    1 132 79 135 87 0
    로 137 79 146 87 0
    ( 23 66 26 75 0
    A 28 66 36 76 0
    S 38 66 47 75 0
    H 49 66 57 76 0
    L 62 67 74 75 0
    E 74 67 78 74 0
    E 83 66 99 76 0
    1 106 65 108 76 0
    8 108 65 117 76 0
    % 117 65 121 76 0
    ) 127 66 130 75 0
    f 196 60 202 62 0
    e 282 0 282 0 0
    . 239 58 251 60 0
    2 94 37 96 44 0
    0 96 36 104 44 0
    2 104 36 105 44 0
    0 108 37 111 44 0
    . 113 36 118 44 0
    0 122 35 127 51 0
    8 127 35 130 51 0
    . 132 36 141 45 0
    1 141 36 141 38 0
    6 146 35 149 44 0
    . 150 36 159 45 0
    A 202 43 210 50 0
    a 224 36 249 50 0
    서 59 20 69 32 0
    울 72 20 84 32 0
    특 85 20 97 32 0
    별 99 20 110 32 0
    시 115 20 123 32 0
    로 132 15 227 54 0
    ! 227 15 237 54 0
    



```python
# image_to_data
# 식별된 text와 좌표정보가 같이 출력됨
data = pytesseract.pytesseract.image_to_data(img, lang="kor+eng", config = config)   
print(data)
# head와 함께 다양한정보를 담음
# 좌표값과 confidence level등을 담고 있어 활용도가 높음.
# 추후 text detection에서 해당 방법을 사용함.
```

    level	page_num	block_num	par_num	line_num	word_num	left	top	width	height	conf	text
    1	1	0	0	0	0	0	0	282	179	-1	
    2	1	1	0	0	0	22	18	229	146	-1	
    3	1	1	1	0	0	22	18	229	146	-1	
    4	1	1	1	1	0	42	18	99	16	-1	
    5	1	1	1	1	1	42	18	99	16	74	주민등록증
    4	1	1	1	2	0	29	47	108	14	-1	
    5	1	1	1	2	1	29	47	14	14	85	홍
    5	1	1	1	2	2	45	47	13	14	91	길
    5	1	1	1	2	3	61	48	13	13	93	동
    5	1	1	1	2	4	77	48	4	12	90	(
    5	1	1	1	2	5	85	48	13	12	51	래
    5	1	1	1	2	6	101	47	13	13	71	좀
    5	1	1	1	2	7	116	47	21	14	50	0)
    4	1	1	1	3	0	28	71	109	10	-1	
    5	1	1	1	3	1	28	71	109	10	88	800101-2345678
    4	1	1	1	4	0	22	91	124	10	-1	
    5	1	1	1	4	1	22	91	8	10	91	서
    5	1	1	1	4	2	33	91	9	10	93	울
    5	1	1	1	4	3	43	91	9	10	3	특
    5	1	1	1	4	4	54	91	18	10	91	별시
    5	1	1	1	4	5	80	93	5	6	93	가
    5	1	1	1	4	6	85	91	14	10	93	산
    5	1	1	1	4	7	101	91	7	10	90	디
    5	1	1	1	4	8	111	91	8	10	93	지
    5	1	1	1	4	9	121	91	9	10	92	털
    5	1	1	1	4	10	132	92	3	8	91	1
    5	1	1	1	4	11	137	92	9	8	84	로
    4	1	1	1	5	0	23	103	228	18	-1	
    5	1	1	1	5	1	23	103	76	10	0	(ASHLEE
    5	1	1	1	5	2	106	103	24	11	61	18%)
    5	1	1	1	5	3	196	117	55	4	23	fe.
    4	1	1	1	6	0	94	128	155	16	-1	
    5	1	1	1	6	1	94	128	65	16	44	2020.08.16.
    5	1	1	1	6	2	202	129	47	14	34	Aa
    4	1	1	1	7	0	59	125	178	39	-1	
    5	1	1	1	7	1	59	147	10	12	93	서
    5	1	1	1	7	2	72	147	12	12	92	울
    5	1	1	1	7	3	85	147	38	12	95	특별시
    5	1	1	1	7	4	132	125	105	39	32	로!
    


# Bounding Box 

### by image_to_boxes


```python
img = cv2.imread('/content/drive/MyDrive/project_03/data/id.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

# print(img.shape)
hImg,wImg,_ = img.shape  # 차원 주의
config = r'--oem 3 --psm 6 outputbase digits' 

########################### image_to_boxes ################################
boxes = pytesseract.pytesseract.image_to_boxes(img, lang='kor+eng', config=config)  
# print(boxes)

for b in boxes.splitlines(): # 한 줄 씩 split
    # print(b)
    b = b.split(' ')  # 나눠서 list에 담아줌
    # print(b)
    x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
    cv2.rectangle(img, (x,hImg-y),(w,hImg-h),(0,255,0), 2)  # 좌하단, 우상단


cv2_imshow(img)
```


![png](text_detection_tesseract_files/text_detection_tesseract_18_0.png)


### by image_to_data


```python
img = cv2.imread('/content/drive/MyDrive/project_03/data/id.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

############################## image_to_data ##########################################
boxes = pytesseract.pytesseract.image_to_string(img, lang='kor+eng')  # 모든 text 
boxes_num = pytesseract.pytesseract.image_to_data(img, lang='kor+eng', config=config)  # 숫자만
# print(boxes)  # str
# print(boxes_num)  # data


for idx, b in enumerate(boxes_num.splitlines()): # 한 줄 씩 split
    '''
    숫자를 가리는 알고리즘 
    '''
    # print(b)
    if idx != 0:  # head를 제외하고 split
        b = b.split()
        # print(b)
        if len(b) ==12:  # 객체가 있는것만 뽑아서
            if len(b[11]) > 13:  # 보통 주민등록번호는 len 13 길다 : 추출한 객체가 11개 이상의 len이면
            # if float(b[10]) > 88 :  # confidence level 높은것, 즉 숫자로 인식한거 중 진짜 숫자인 것만 가리기
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단
               


cv2_imshow(img)
```


![png](text_detection_tesseract_files/text_detection_tesseract_20_0.png)



```python
%%shell
jupyter nbconvert --to markdown /content/text_detection_tesseract.ipynb
```
