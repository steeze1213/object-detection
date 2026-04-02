# 비전 시스템 & 딥러닝

---

## YOLOv3 Object Detection

YOLOv3 사전 학습 모델과 OpenCV를 활용한 객체 검출 프로그램

---

### 개요

| 항목 | 내용 |
|------|------|
| 모델 | YOLOv3 (COCO 데이터셋 사전 학습) |
| 탐지 클래스 | 80가지 (사람, 자동차, 동물 등) |
| 입력 | 정지 이미지 / 웹캠 실시간 영상 |
| 신뢰도 임계값 | 0.5 이상만 탐지 결과로 표시 |

---

### 환경

- Python 3.9
- opencv-python
- numpy
``` powershell
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