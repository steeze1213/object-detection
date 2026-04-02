from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO("fruits/last.pt")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    while True:
        delay = int(cap.get(cv2.CAP_PROP_FPS))

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

                    if confidence > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{label} {confidence:.2f}",
                                    (x1, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("stream", img)
            key = cv2.waitKey(delay)
            if key == 27:
                break

cap.release()
cv2.destroyAllWindows()