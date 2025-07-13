import cv2
import mediapipe as mp

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands # модуль MediaPipe, отвечающий за работу с руками
hands = mp_hands.Hands() # Экземпляр детектора рук с дефолтными параметрами
mp_draw = mp.solutions.drawing_utils  # Для визуализации точек

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0) # Выводим камеру
cap.set(3, wCam)
cap.set(4, hCam)

cv2.namedWindow("Result Hand", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result Hand", wCam, hCam)

while True:
    success, img = cap.read()

    # Конвертация в RGB 
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Если руки обнаружены, рисуем точки
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(5, 92, 7)),  
                mp_draw.DrawingSpec(color=(0, 255, 102))
            )

    cv2.imshow("Result Hand", img)
    
    # На q завершаем цикл
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()