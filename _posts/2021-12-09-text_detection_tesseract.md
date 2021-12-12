---
layout: post
title:  "[project3-1] idcard deidentification"
---

# tesseract를 통해 이미지를 식별하고 bbox를 사용하여 주요정보를 가리는 것을 연습합니다.



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

### 한글깨짐 방지
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
    5	1	1	1	1	1	42	18	99	16	74	주민등록증
    4	1	1	1	2	0	29	47	108	14	-1	
    5	1	1	1	2	1	29	47	14	14	85	홍
    5	1	1	1	2	2	45	47	13	14	91	길
    5	1	1	1	2	3	61	48	13	13	93	동
    5	1	1	1	3	1	28	71	109	10	88	800101-2345678


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
![image](https://user-images.githubusercontent.com/86705085/145714853-a1a4184d-c5e8-4cba-8e20-c39cb3022be8.png)


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

![image](https://user-images.githubusercontent.com/86705085/145714922-4c5e98a3-5bed-4b99-bf32-25cd08bf9805.png)
