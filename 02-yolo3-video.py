import cv2
import numpy as np
import os

# 현재 스크립트 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))

# 사전 학습된 YOLO 모델 읽어오기
net = cv2.dnn.readNet(
    os.path.join(base_dir, "yolov3.weights"),
    os.path.join(base_dir, "yolov3.cfg")
)

# 클래스명 리스트 생성 및 채우기
classes = []
with open(os.path.join(base_dir, "sample.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 출력층 따로 구분해서 리스트화
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1]
                 for i in net.getUnconnectedOutLayers()]

# 객체 표시할 색상 랜덤하게 생성하기
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 웹캠 열기 실패 시 즉시 종료
if not cap.isOpened():
    print("webcam error")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        print("frame error")
        break

    height, width, channels = frame.shape

    # 검출 수행
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 검출 결과 파싱
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 중복 박스 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 결과 화면에 그리기
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]  # 클래스 기준으로 색상 선택
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

    cv2.imshow("YOLOv3 Real-time Detection", frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

"""

# 02-yolo3-video.py
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("sample.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if cap.isOpened():
    while True:

        # 현재 카메라의 초당 프레임수
        delay = int(cap.get(cv2.CAP_PROP_FPS))

        ret, img = cap.read()

        if ret:

            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # 좌표
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

            cv2.imshow("stream", img)
            key = cv2.waitKey(delay)
            if key == 27:
                # if you push the ESC key,
                break

cv2.destroyAllWindows()
"""