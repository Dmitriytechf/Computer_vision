import cv2
import mediapipe as mp
import time


# Инициализация детектора лиц людей 
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=1)

pTime = 0


wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0) # Вебка
cap.set(3, wCam)
cap.set(4, hCam)

# Задаем размер окна
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", wCam, hCam)
while True:
    success, img = cap.read()
    if not success:
        print('Программа завершилась')
        break

    # Конвертация в RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = face_detector.process(img_rgb)
    person_count = 0

    if results.detections:
        for detection in results.detections:
            person_count += 1
            # Получение координат лица
            bbox = detection.location_data.relative_bounding_box # Получаем отсносительные координаты лица
            # Переводим в абсолютные координаты
            h, w, _ = img.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            # Увеличивае квадрат вокруг лица
            expand = 20 
            x = max(0, x - expand)
            y = max(0, y - expand)
            width = min(w - x, width + 2 * expand)
            height = min(h - y, height + 2 * expand)

            # Квадра вокруг лица
            cv2.rectangle(img, (x, y), (x+width, y+height), (0, 255, 0), 5)
    
    # Количество людей надпись
    cv2.putText(img, f'People: {person_count}', (450, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    
    # Расчет ФПС
    cTime = time.time() 
    fps = 1 / (cTime - pTime) 
    pTime = cTime
    
    cv2.putText(img, f'FPS {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Result", img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        print('Программа завершила свою работу')
        break

cap.release()
cv2.destroyAllWindows()