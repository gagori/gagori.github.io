---
layout: post
title:  "[project3-6-3] Face Detectin with Haar Cascade"

---

# Face Detection
이전 포스트에서는 YOLO v3/ v5 로 Face Detection을 진행하였다. 
고전적인 방법으로 haar 특징자를 이용한 cascade 모델을 가지고 얼굴을 인식하는 방법이 존재한다고 한다.


그렇다면 이번 포스트에선 cascade사용법과 결과물/문제점에 대해 살펴보고자 한다.




### -----------------------------------------------------------------------------------
## Haar Cascade

Haar Cascade는 머신러닝 기반의 object detection 알고리즘이다.

haar 특징을 기반으로 영상에서 오브젝트를 검출하기 위해 사용된다.
직사각형 영역으로 구성되는 특징을 사용하기 때문에 픽셀을 직접 사용할 때 보다 동작 속도가 빠르다고 한다.

찾으려는 오브젝트(fontal face이므로 얼굴)가 포함된 이미지와 오브젝트가 없는 이미지를 사용하여 하르 특징 분류기를 학습시키고 이후 검출을 진행하는 과정이다.



알고리즘은 4단계로 구성된다.
1. Haar Feature Selection
2. Creating Integral Images
3. Adaboost Training
4. Cascading Classifiers


<Haar Feature Selection>
 
![image](https://user-images.githubusercontent.com/86705085/147210119-04ee39b3-2465-4270-95ba-387bb5abf66e.png)

![image](https://user-images.githubusercontent.com/86705085/147210197-354be573-cfb0-426a-a76b-6d5c4c7b9895.png)



### 1. 모델 불러오기
cascade classifer는 open cv에서 쉽게 불러올 수 있다는 장점이 잇따.
추가로 우리는 face를 찾을것이기 때문에 haar정보가 담겨있는 xml파일을 다운받아야 한다.
```python
cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```


### 2. Object Detection
본격적으로 face detection을 위해 model에 해당 이미지를 넣고, 커널사이즈를 조절하였다.
```python
results=cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(5,5)) # 숫자 바꿔보기!! 낮추니까 얼굴을 더 잘잡았음.
```
results를 확인해보면 좌표값이 담겨있음을 알 수 있다.
주의할 점은 (좌상단/ 우하단)이 아니라 (좌상단/넓이,높이) 의 구성을 가지고 있다는 점이다.


### 3.  정보 표시

results에 담겨있는 좌표정보를 활용하여 이미지에 정보를 표시한다.
```python
for b in results:
	print(b)
	x,y,w,h = b
	cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), -1)
	cv2.putText(img, "haar", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)
```

### 4. 결과 및 문제점

Haar Cascade로 face detection을 한 결과이다.
대부분의 경우 얼굴인식이 잘 되었지만 아래와 같이 문제가 있는 경우도 있었다.
보이는 것처럼 중구난방으로 얼굴을 잡는 문제가 종종 발생했다.


![image](https://user-images.githubusercontent.com/86705085/147211548-09b4c6a9-0bde-4647-8cdc-73e49a2d9f00.png)






이전 포스트에서 yolo를 가지고 detection을 진행했는데
똑같은 이미지를 가져와서 결과를 비교해보았다.

haar Cascade와 달리 중구난방으로 객체를 잡는 문제가 해결되었다.
따라서 향후 프로젝트에서는 yolo를 통해 face detection을 진행하기로 한다.
![image](https://user-images.githubusercontent.com/86705085/147211395-b90679e8-7859-4d09-91f4-bb6b60472cfe.png)













**[참고자료]**
 - https://webnautes.tistory.com/1352


