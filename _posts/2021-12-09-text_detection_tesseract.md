

### tesseract 설치
  - https://velog.io/@latte_h/Tesseract


```python
from google.colab import drive
drive.mount("/content/drive")
```

    Mounted at /content/drive



```python
!sudo apt install tesseract-ocr
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following additional packages will be installed:
      tesseract-ocr-eng tesseract-ocr-osd
    The following NEW packages will be installed:
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd
    0 upgraded, 3 newly installed, 0 to remove and 37 not upgraded.
    Need to get 4,795 kB of archives.
    After this operation, 15.8 MB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-eng all 4.00~git24-0e00fe6-1.2 [1,588 kB]
    Get:2 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-osd all 4.00~git24-0e00fe6-1.2 [2,989 kB]
    Get:3 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr amd64 4.00~git2288-10f4998a-2 [218 kB]
    Fetched 4,795 kB in 1s (4,384 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 3.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package tesseract-ocr-eng.
    (Reading database ... 155222 files and directories currently installed.)
    Preparing to unpack .../tesseract-ocr-eng_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-eng (4.00~git24-0e00fe6-1.2) ...
    Selecting previously unselected package tesseract-ocr-osd.
    Preparing to unpack .../tesseract-ocr-osd_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-osd (4.00~git24-0e00fe6-1.2) ...
    Selecting previously unselected package tesseract-ocr.
    Preparing to unpack .../tesseract-ocr_4.00~git2288-10f4998a-2_amd64.deb ...
    Unpacking tesseract-ocr (4.00~git2288-10f4998a-2) ...
    Setting up tesseract-ocr-osd (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr-eng (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr (4.00~git2288-10f4998a-2) ...
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...



```python
!pip install pillow
```

    Requirement already satisfied: pillow in /usr/local/lib/python3.7/dist-packages (7.1.2)



```python
!pip install pytesseract
```

    Collecting pytesseract
      Downloading pytesseract-0.3.8.tar.gz (14 kB)
    Requirement already satisfied: Pillow in /usr/local/lib/python3.7/dist-packages (from pytesseract) (7.1.2)
    Building wheels for collected packages: pytesseract
      Building wheel for pytesseract (setup.py) ... [?25l[?25hdone
      Created wheel for pytesseract: filename=pytesseract-0.3.8-py2.py3-none-any.whl size=14072 sha256=d899d837d909632dd6f103f687cb585bfceccdb846216785a2fd450355425b73
      Stored in directory: /root/.cache/pip/wheels/a4/89/b9/3f11250225d0f90e5454fcc30fd1b7208db226850715aa9ace
    Successfully built pytesseract
    Installing collected packages: pytesseract
    Successfully installed pytesseract-0.3.8


### 필요한 라이브러리 다운받기


```python
from PIL import Image
import pytesseract
import cv2
from google.colab.patches import cv2_imshow

```

# pytesseract 연습
  - image_to_string
  - image_to_data
  - image_to_boxes


```python
img = Image.open('/content/drive/MyDrive/project_03/data/id.jpg')        # 이미지 오픈시 pillow써서 open, 우리는 cv2로 쓸듯
data = pytesseract.pytesseract.image_to_string(img, lang="kor+eng")   # 숫자는 잘 인식하지만 한글은 잘 인식 못함.
# image_to_data    데이터형식
# image_to_boxes   행렬 데이터로
# image_to_string  txt 그대로

print(data)

# with open("sample.txt", "w") as f:   # 현재줌치에 txt 파일 만들어줌
#     f.write(text)
```

    주민등록증
    홍길동(래좀0)
    
    800101-2345678
    
    Alesana eS
    (대틈테크노타운 18차)
    
     
    
    2020.08.16.
    서울특별시 금천구청장|
    


#### 한글깨짐 방지
  - https://webnautes.tistory.com/947


```python
!sudo apt install tesseract-ocr-kor
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      tesseract-ocr-kor
    0 upgraded, 1 newly installed, 0 to remove and 37 not upgraded.
    Need to get 1,050 kB of archives.
    After this operation, 1,693 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-kor all 4.00~git24-0e00fe6-1.2 [1,050 kB]
    Fetched 1,050 kB in 1s (1,320 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package tesseract-ocr-kor.
    (Reading database ... 155269 files and directories currently installed.)
    Preparing to unpack .../tesseract-ocr-kor_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-kor (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr-kor (4.00~git24-0e00fe6-1.2) ...



```python
!sudo apt install tesseract-ocr-script-hang
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      tesseract-ocr-script-hang
    0 upgraded, 1 newly installed, 0 to remove and 37 not upgraded.
    Need to get 1,854 kB of archives.
    After this operation, 4,861 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 tesseract-ocr-script-hang all 4.00~git24-0e00fe6-1.2 [1,854 kB]
    Fetched 1,854 kB in 1s (2,197 kB/s)
    debconf: unable to initialize frontend: Dialog
    debconf: (No usable dialog-like program is installed, so the dialog based frontend cannot be used. at /usr/share/perl5/Debconf/FrontEnd/Dialog.pm line 76, <> line 1.)
    debconf: falling back to frontend: Readline
    debconf: unable to initialize frontend: Readline
    debconf: (This frontend requires a controlling tty.)
    debconf: falling back to frontend: Teletype
    dpkg-preconfigure: unable to re-open stdin: 
    Selecting previously unselected package tesseract-ocr-script-hang.
    (Reading database ... 155273 files and directories currently installed.)
    Preparing to unpack .../tesseract-ocr-script-hang_4.00~git24-0e00fe6-1.2_all.deb ...
    Unpacking tesseract-ocr-script-hang (4.00~git24-0e00fe6-1.2) ...
    Setting up tesseract-ocr-script-hang (4.00~git24-0e00fe6-1.2) ...



```python
img = Image.open('/content/drive/MyDrive/project_03/data/id.jpg') 
# data = pytesseract.pytesseract.image_to_string(img, lang="kor+eng")   # 방법2
data = pytesseract.pytesseract.image_to_string(img, lang="Hangul")    # 방법1

print(data)
```

    주민등록증
    홍 길 동 ( 빠 솜 0)
    
    800101-2345678
    
    서 울 특별시 가 산 디 지 털 1 로
    ( 대 륭 데 크 노 타운 18 차 )
    
     
    
    2020.08.16.
    서 올 특별시 금 천 구 청 장 |
    


#### 숫자만 config 옵션


```python
config = r'--oem 3 --psm 6 outputbase digits'
img = Image.open('/content/drive/MyDrive/project_03/data/id.jpg')        # 이미지 오픈시 pillow써서 open, 우리는 cv2로 쓸듯
data = pytesseract.pytesseract.image_to_string(img, lang="kor+eng", config = config)   # 숫자는 잘 인식하지만 한글은 잘 인식 못함.

print(data)
```

    주민등록증
    홍길동(래좀0)
    800101-2345678
    서울특별시 가산디지털1로
    (ASHLEE 18%)         fe.
    2020.08.16.   Aa
    서울특별시 로!
    


# 실습, 사진띄우기


```python
import cv2
import pytesseract


# img = Image.open('/content/drive/MyDrive/project_03/data/id.jpg') 
img = cv2.imread('/content/drive/MyDrive/project_03/data/id.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
# cv2_imshow(img)

# print(img.shape)
hImg,wImg,_ = img.shape
config = r'--oem 3 --psm 6 outputbase digits' # 숫자만 

# ########################### image_to_boxes ################################
# boxes = pytesseract.pytesseract.image_to_boxes(img, lang='kor+eng', config=config)  # 숫자만
# # print(boxes)

# for b in boxes.splitlines(): # 한 줄 씩 split
#     # print(b)
#     b = b.split(' ')  # 나눠서 list에 담아줌
#     # print(b)
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(img, (x,hImg-y),(w,hImg-h),(0,255,0), 2)  # 좌하단 우상단...?

# cv2_imshow(img)


############################## image_to_data ##########################################
############################## image_to_string ########################################
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
            # if len(b[11]) > 11:  # 보통 주민등록번호는 len 13 길다 : 추출한 객체가 11개 이상의 len이면
            if float(b[10]) > 88 :  # confidence level 높은것, 즉 숫자로 인식한거 중 진짜 숫자인 것만 가리기
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0), -1)   # 좌상단, 우하단
                cv2.putText(img, b[11], (x,y+25), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0,0,255), thickness=1)


boxes_list = []
for b in boxes.splitlines():
    '''
    인식된 text를 모두 리스트에 담아, 
    필요한 정보만 indexing 할 수 있는 알고리즘 
    '''
    # print(b)
    boxes_list.append(b)  # list에 담으면 indexing으로 원하는 것만 가져오기 편하므로
print(boxes_list)  # 글자 깨짐. 딥러닝으로 pretrained된 모델( frozen_east_text_detection )을 가져와서 쓰면 잘 인식함.

cv2_imshow(img)




```

    ['주민등록증', 'SAS (KEM)', '800101-2345678', '', 'Aesan Age', '(대틈테크노타운 18차)', '', ' ', '', '2020.08.16.', '서울특별시 금천구청장|', '']



![png](text_detection_tesseract_files/text_detection_tesseract_16_1.png)



```python
# print(boxes_num)
```


```python

```

# preprocessing practice


```python
import cv2
import tempfile
import numpy as np
from PIL import Image

def ShowImage(_ViewImg, _Title="Image View"):
    #DEF.FUNCNAME()
    cv2_imshow(_ViewImg)
    # while True:
    #     nKeyCode = cv2.waitKey(1)
    #     if nKeyCode == ord('x') or nKeyCode == ord('X'):
    #         cv2.destroyAllWindows()
    #         break
    #     elif nKeyCode == ord('n') or nKeyCode == ord('N'):
    #         break
        
def LoadImageAndNameEncoding(_ImageFileName):
    # 이미지 이름이 한글일경우 못읽는 문제가 windows에서 발생
    # 한글명일경우 (영문일지라도) decoding을 해줌.
    # return cv2 type
    stream = open(_ImageFileName.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    Img = cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)
    ShowImage(Img, "(LoadImageAndNameEncoding) OriginalImage")
    return Img

def ImageAutoResize(_FilePath):
    # img_w = 1000 #OCR 이미지의 가로길이을 설정

    # ocr 을 하기 좋은 상태의 이미지 크기로 조정한다.
    # im = cv2.imread(file_path)
    Load_Img = LoadImageAndNameEncoding(_FilePath)

    # 이미지크기 auto 조정
    fRatio_Width = 1
    fRatio_Height = 1
    nZoom_Width = 0
    nZoom_Height = 0
    nScalePercent = 100
    fExt = 1.2 # 확대할 크기를 구함.

    nHeight, nWidth = Load_Img.shape[:2]
    fRatio_Width = float(nWidth / nHeight)
    #DEF.DEBUG_PRINT("=> (Width/Height):%d / %d, Ratio(Width/Height):%f/%f,  " %(nWidth,nHeight,fRatio_Width, fRatio_Height))

    # 확대 또는 축소를 결정 1보다 크면 축소 1보다 작으면 확대
    fRadio_Zoom = float(nWidth / G_IMAGE_SIZE_WIDTH)  # 아래서 1000으로 설정해둠

    if fRadio_Zoom >= 1:  # 축소를 수행
        nZoom_Width = round(nWidth / fRadio_Zoom)  # 축소할 가로 길이를 구함
        nZoom_Heigth = round(nZoom_Width / fRatio_Width)  # 축소할 세로 길이를 구함.
    elif fRadio_Zoom < 1:  # 확대를 수행
        fExt = float(fExt / fRadio_Zoom)  # 확대할 크기를 구함.
        nZoom_Width = round(nWidth * fExt)  # 확대할 가로 길이를 구함
        nZoom_Heigth = round(nZoom_Width / fRatio_Width)  # 확대할 세로 길이를 구함.


    #print("=> (Radio_Zoom):%f (Ext:%f)=>  (Width/Height):%d/%d  " %(fRadio_Zoom, fExt, nZoom_Width, nZoom_Heigth))
    # 이미지 크기 변경
    dSize = (int(nZoom_Width), int(nZoom_Heigth))
    # print(dim)
    Load_Img = cv2.resize(Load_Img, dSize, interpolation=cv2.INTER_AREA)
    # cv2 image to change  PIL Image
    brg2rgb_Img = cv2.cvtColor(Load_Img, cv2.COLOR_BGR2RGB)
    Arrypil_Img = Image.fromarray(brg2rgb_Img)
    ShowImage(brg2rgb_Img, "(ImageAutoResize) Load_Img")
    ShowImage(brg2rgb_Img, "(ImageAutoResize) brg2rgb_Img")
#    ShowImage(arrypil_Img, "arrypil_Img")
    return Arrypil_Img


# 세로이미지가 더큰경우에 대한 이미지 크기 자동화
def ImageAutoResize_Height( _FilePath):
    # img_w = 1000 #OCR 이미지의 가로길이을 설정

    # ocr 을 하기 좋은 상태의 이미지 크기로 조정한다.
    # im = cv2.imread(file_path)
    Load_Img = LoadImageAndNameEncoding(_FilePath)
    ShowImage(Load_Img, "(ImageAutoResize_Height) LoadImageAndNameEncoding")
    # 이미지크기 수동 조정
    nHeight, nWidth = Load_Img.shape[:2]

    nRatio_Width = 1
    fRatio_Height = float(nHeight / nWidth)
    nZoom_Width = 0
    nZoom_Height = 0

    nDef_Width = G_IMAGE_SIZE_WIDTH

    fRatio_Zoom= float(nHeight / nDef_Width)  # 확대 또는 축소를 결정 1보다 크면 축소 1보다 작으면 확대

    print("=> Ratio_Zoom : %f" %(fRatio_Zoom))
    if fRatio_Zoom >= 1:  # 축소를 수행
        zoom_heigth = round(nHeight / fRatio_Zoom)  # 축소할 세로 길이를 구함.
        nZoom_Width = round(zoom_heigth / fRatio_Height)  # 축소할 가로 길이를 구함

    elif fRatio_Zoom < 1:  # 확대를 수행
        ext = float(0.7 / fRatio_Zoom)  # 확대할 크기를 구함.
        zoom_heigth = round(nHeight * ext)  # 확대할 세로 길이를 구함.
        nZoom_Width = round(zoom_heigth / fRatio_Height)  # 확대할 가로 길이를 구함

    dim = (int(zoom_heigth), int(zoom_heigth))
    # dim = (632, 1000)
    # print(dim)
    Load_Img = cv2.resize(Load_Img, dim, interpolation=cv2.INTER_AREA)

    # img_show(im)

    # cv2 image to change  PIL Image
    pil_image = cv2.cvtColor(Load_Img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pil_image)
    return im_pil

# 수동으로 이미지 변경
def ImageManualResize(_FilePath):
    # img_w = 1000 #OCR 이미지의 가로길이을 설정

    # ocr 을 하기 좋은 상태의 이미지 크기로 조정한다.
    # im = cv2.imread(file_path)
    im = LoadImageAndNameEncoding(_FilePath)
    # 이미지크기 auto 조정
    scale_percent = 100
    height, width = im.shape[:2]

    ratio_w = float(width / height)
    ratio_h = 1
    zoom_width = 0
    zoom_height = 0

    zoom = float(width / (G_IMAGE_SIZE_WIDTH))  # 확대 또는 축소를 결정 1보다 크면 축소 1보다 작으면 확대

    if zoom >= 1:  # 축소를 수행
        zoom_width = round(width / zoom)  # 축소할 가로 길이를 구함
        zoom_heigth = round(zoom_width / ratio_w)  # 축소할 세로 길이를 구함.
    elif zoom < 1:  # 확대를 수행
        ext = float(1.2 / zoom)  # 확대할 크기를 구함.
        zoom_width = round(width * ext)  # 확대할 가로 길이를 구함
        zoom_heigth = round(zoom_width / ratio_w)  # 확대할 세로 길이를 구함.

    # 이미지 크기 변경
    dim = (int(zoom_width), int(zoom_heigth))
    # print(dim)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    # cv2 image to change  PIL Image
    pil_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pil_image)
    return im_pil

# 수동으로 이미지 변경 세로이미지가 더큰경우
def ImageManualResize_Height(file_path):
    # img_w = 1000 #OCR 이미지의 가로길이을 설정

    # ocr 을 하기 좋은 상태의 이미지 크기로 조정한다.
    # im = cv2.imread(file_path)
    im = LoadImageAndNameEncoding(file_path)
    # 이미지크기 수동 조정
    height, width = im.shape[:2]

    ratio_w = 1
    ratio_h = float(height / width)
    zoom_width = 0
    zoom_height = 0

    c_width = G_IMAGE_SIZE_WIDTH * G_IMAGE_SIZE

    zoom = float(height / c_width)  # 확대 또는 축소를 결정 1보다 크면 축소 1보다 작으면 확대

    print(zoom)
    if zoom >= 1:  # 축소를 수행
        zoom_heigth = round(height / zoom)  # 축소할 세로 길이를 구함.
        zoom_width = round(zoom_heigth / ratio_h)  # 축소할 가로 길이를 구함

    elif zoom < 1:  # 확대를 수행
        ext = float(0.7 / zoom)  # 확대할 크기를 구함.
        zoom_heigth = round(height * ext)  # 확대할 세로 길이를 구함.
        zoom_width = round(zoom_heigth / ratio_h)  # 확대할 가로 길이를 구함

    dim = (int(zoom_heigth), int(zoom_heigth))
    # dim = (632, 1000)
    # print(dim)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

    # img_show(im)

    # cv2 image to change  PIL Image
    pil_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(pil_image)
    return im_pil

def LoadImageAndNameEncodingAndBGR2RGB(_FileName):
    StreamData = open(_FileName.encode("utf-8"), "rb")
    ArrByteData = bytearray(StreamData.read())
    numpyArray = np.asarray(ArrByteData, dtype=np.uint8)
    Img = cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)

    # pil 형태로 변경
    brg2rgb_Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    ShowImage(brg2rgb_Img, "brg2rgb_Img")
    im_pil = Image.fromarray(brg2rgb_Img)
    return im_pil

def SetImageDpi(_FilePath):
    # DEF.FUNCNAME()
    im = LoadImageAndNameEncodingAndBGR2RGB(_FilePath)
    # im = Image.open(file_path)

    # DEF.DEBUG_PRINT("이미지 포켓을 변경 (png,gif,tiff.pcx,bmp,jpg)")
    # 이미지 포켓을 변경 (png,gif,tiff.pcx,bmp,jpg)
    rgb_im = im.convert('RGB')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    rgb_im.save(temp_filename)

    # DEF.DEBUG_PRINT("이미지 사이즈변경")
    # 이미지 사이즈변경
    length_x, width_y = im.size
    img_size = G_IMAGE_SIZE
    size = (round(length_x * img_size), round(width_y * img_size))
    im_resized = rgb_im.resize(size, Image.ANTIALIAS)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name

    # DEF.DEBUG_PRINT("해상도 변경")
    # 해상도 변경
    im_resized.save(temp_filename, dpi=(G_DPI, G_DPI))
    # im_resized.save('d:\\result.jpg',dpi=(DPI, DPI))
    #ShowImage(im_resized, "SetDPI");
    return temp_filename

def image_smoothening(img,_ThredholdCount):
        ret1, th1 = cv2.threshold(img, G_BINARY_THREHOLD+(G_BINARY_THREHOLD_ALPHA*_ThredholdCount), 255, cv2.THRESH_BINARY)
        ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(th2, (1, 1), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th3

def RemoveNoiseAndSmooth(file_name,_ThredholdCount):
        img = cv2.imread(file_name, 0)
        filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img = image_smoothening(img,_ThredholdCount)
        or_image = cv2.bitwise_or(img, closing)
        return or_image

def ProcessImageForOcr(_FilePath,_ThredholdCount):
    temp_filename = SetImageDpi(_FilePath)
    im_new = RemoveNoiseAndSmooth(temp_filename,_ThredholdCount)
    return im_new

# =========================
imgfile='/content/drive/MyDrive/project_03/data/id.jpg'
# G_DPI = int(_Argv[10]) if int(_Argv[10]) >=0 else 300
G_DPI = 300
G_IMAGE_AUTO = 'ON'
G_IMAGE_SIZE_WIDTH = 1000
G_IMAGE_SIZE = 1
# G_BINARY_THREHOLD           = int(_Argv[5]) if int(_Argv[5]) >0 else 120  # 이진화 경계값
G_BINARY_THREHOLD = 120
# G_BINARY_THREHOLD_ALPHA     = int(_Argv[6]) if int(_Argv[6]) >=0 else 0 #이진화 간격
G_BINARY_THREHOLD_ALPHA = 0
# G_BINARY_THREHOLD_BETA      = int(_Argv[7]) if int(_Argv[7]) >=0 else 0 #이진화 최종
G_BINARY_THREHOLD_BETA = 0
# =========================


Ori_Img = cv2.imread(imgfile)
nHeight, nWidth = Ori_Img.shape[:2]

#print("=> ORIGINAL IMG > Height:%d, Width:%d" %(nHeight, nWidth))
bgr2rgb_Img = cv2.cvtColor(Ori_Img, cv2.COLOR_BGR2RGB)
ShowImage(bgr2rgb_Img, "(ImagePreprocessing) bgr2rgb_Img")

ArrayPil_Img = Image.fromarray(bgr2rgb_Img)
#if DEF._DEF_USING_DEBUG == 1:    ShowImage(im_pil, "fromarray");
TempFile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#temp_filename = TempFile.name
ArrayPil_Img.save(TempFile.name, dpi=(G_DPI, G_DPI))
#print("=> temp_filename:%s" %(TempFile.name))

#ShowImage(im_pil, "fromarray");
Resize_Img = ''
#print("=> AutoResize:" + DEF.G_IMAGE_AUTO)
if G_IMAGE_AUTO == 'ON':  # 이미지 자동 사이즈 조정
    if nWidth >= nHeight:
        # 가로이미지가 더큰경우 (일반적으로)
        Resize_Img = ImageAutoResize(TempFile.name)  # 이미지 사이즈 변경
    else:
        # 세로이미자가 더큰경우 (드물지만)
        Resize_Img = ImageAutoResize_Height(TempFile.name)  # 이미지 사이즈 변경
else:  # 이미지 수동 사이즈 조정
    if nWidth >= nHeight:
        # 가로이미지가 더큰경우 (일반적으로)
        Resize_Img = ImageManualResize(TempFile.name)  # 이미지 사이즈 변경
    else:
        # 세로이미자가 더큰경우 (드물지만)
        Resize_Img = ImageManualResize_Height(TempFile.name)  # 이미지 사이즈 변경

nWidth, nHeight = Resize_Img.size
# 축소또는 확되된 이미지의 width,height 의 정보를 서버로 보낸다.
strResizeInfo = str(nWidth) + ':' + str(nHeight)
# ws_send("work_image_info", work_width_height)
#print("=> Resize Image > %s" %(strResizeInfo))

# 이미지 포켓을 jpg로 변경 (png,gif,tiff.pcx,bmp,jpg)
rgb_Img = Resize_Img.convert('RGB')
TempFile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
#temp_filename = TempFile.name
rgb_Img.save(TempFile.name)
Ori_Img = cv2.imread(TempFile.name, 1)
# 확대 또는 축소되어있고 이진화 및 기타 각종 전처리 되어있는 이미지
Working_Img = ProcessImageForOcr(TempFile.name, 0)
ShowImage(Ori_Img, "(ImagePreprocessing) Original_Image")
ShowImage(Working_Img, "(ImagePreprocessing) Working_Image")

# Ori_Img, Working_Img
```


```python

```


```python

```


```python

```


```python

```


```python
cv2.imshow()
```


    ---------------------------------------------------------------------------

    DisabledFunctionError                     Traceback (most recent call last)

    <ipython-input-43-e7b208bf19ef> in <module>()
    ----> 1 cv2.imshow()
    

    /usr/local/lib/python3.7/dist-packages/google/colab/_import_hooks/_cv2.py in wrapped(*args, **kwargs)
         50   def wrapped(*args, **kwargs):
         51     if not os.environ.get(env_var, False):
    ---> 52       raise DisabledFunctionError(message, name or func.__name__)
         53     return func(*args, **kwargs)
         54 


    DisabledFunctionError: cv2.imshow() is disabled in Colab, because it causes Jupyter sessions
    to crash; see https://github.com/jupyter/notebook/issues/3935.
    As a substitution, consider using
      from google.colab.patches import cv2_imshow




```python

```
