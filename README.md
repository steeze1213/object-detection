# 비전 시스템 & 딥러닝

---

## YOLOv3 Object Detection

YOLOv3 사전 학습 모델과 OpenCV를 활용한 객체 검출 프로그램

---

### 개요

| 항목      | 내용                       |
|---------|--------------------------|
| 모델      | YOLOv3 (COCO 데이터셋 사전 학습) |
| 탐지 클래스  | 80가지 (사람, 자동차, 동물 등)     |
| 입력      | 정지 이미지 / 웹캠 실시간 영상       |
| 신뢰도 임계값 | 0.5 이상만 탐지 결과로 표시        |

---

### 환경

- Python 3.9
- opencv-python
- numpy
```powershell
py -3.9 -m venv venv39
.\venv39\Scripts\activate
pip install opencv-python numpy
```

### 사전 준비

`yolov3.cfg`와 `sample.names`는 저장소에 포함  
`yolov3.weights`는 아래 링크에서 직접 다운로드 후 프로젝트 루트에 위치시킬 것
```
https://pjreddie.com/media/files/yolov3.weights
```

---

### 실행 방법

**정지 이미지 검출**
```powershell
python 01-yolo3-object.py
```

**웹캠 실시간 검출**
```powershell
python 02-yolo3-video.py
```

- 웹캠 실행 중 `ESC` 키를 누르면 종료

---

### 주요 코드

**모델 로드**
```python
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
```

**이미지 전처리 및 추론**
```python
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, False)
net.setInput(blob)
outs = net.forward(output_layers)
```

**중복 박스 제거 (NMS)**
```python
# 신뢰도 0.5 미만 제거, 박스 겹침 비율 0.4 이상 제거
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

---

## YOLOv8 Object Detection

YOLOv8 사전 학습 모델과 OpenCV를 활용한 실시간 객체 검출 프로그램

---

### 개요

| 항목      | 내용                        |
|---------|---------------------------|
| 모델      | YOLOv8m (COCO 데이터셋 사전 학습) |
| 탐지 클래스  | 80가지 (사람, 자동차, 동물 등)      |
| 입력      | 웹캠 실시간 영상                 |
| 신뢰도 임계값 | 0.5 이상만 탐지 결과로 표시         |

---

### 환경

- Python 3.9
- ultralytics
- opencv-python
```powershell
pip install ultralytics opencv-python
```

### 사전 준비

`yolov8m.pt`는 최초 실행 시 자동으로 다운로드됨

---

### 실행 방법

**웹캠 실시간 검출**
```powershell
python 04-yolo8-video.py
```

- 웹캠 실행 중 `ESC` 키를 누르면 종료

---

### 주요 코드

**모델 로드**
```python
model = YOLO("yolov8m.pt")
```

**객체 검출 및 결과 파싱**
```python
results = model(img, verbose=False)

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
```

**결과 시각화**
```python
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
cv2.putText(img, f"{label} {confidence:.2f}",
            (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
```