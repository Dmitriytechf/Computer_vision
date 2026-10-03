import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt") 
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Не удалось открыть камеру")
    exit()

while True:
    okay, frame = camera.read()
    
    if not okay:
        break

    results = model(frame, conf=0.5, verbose=False)
    frame_box = results[0].plot()

    cv2.imshow("CV", frame_box)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("CV", cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()
    