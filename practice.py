from ultralytics import YOLO
import cv2

# 클래스별 색상 지정 (BGR)
CLASS_COLORS = {
    'cat':       (0, 0, 255),    # 빨
    'cow':       (0, 100, 255),  # 주
    'dog':       (0, 255, 255),  # 노
    'sciuridae': (0, 255, 0),    # 초
    'sheep':     (255, 100, 0),  # 파
    'spider':    (255, 0, 255),  # 보
}

# YOLOv8 모델 로드
model = YOLO("runs/detect/animals_yolo/weights/last.pt")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        ret, img = cap.read()

        if ret:
            # 객체 검출 수행
            results = model(img, verbose=False)

            # 검출 결과 그리기
            for result in results:
                for box in result.boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 클래스명 및 신뢰도
                    label = model.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    color = CLASS_COLORS.get(label, (255, 255, 255))

                    if confidence > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, f"{label} {confidence:.2f}",
                                    (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            cv2.imshow("stream", img)
            if cv2.waitKey(1) == 27:
                break

cap.release()
cv2.destroyAllWindows()