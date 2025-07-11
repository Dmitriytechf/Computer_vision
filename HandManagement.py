import cv2
import mediapipe as mp
import pyautogui
import math
import time


# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands 
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) 
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
pTime = 0

cap = cv2.VideoCapture(0)

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 1280, 720)
while True:
    success, img = cap.read()

    # Конвертация в RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Если руки обнаружены, рисуем точки
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 0, 222)),  
                mp_draw.DrawingSpec(color=(0, 255, 102))
            )
            # Указательный палец
            index_finger = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            
            # Преобразуем координаты в пиксели
            h, w, _ = img.shape
            index_x = int(index_finger.x * w)
            index_y = int(index_finger.y * h)
            
            # Управление курсором мыши
            mouse_x = screen_w * (index_finger.x)
            mouse_y = screen_h * (index_finger.y)
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
            
            distance = math.sqrt(
                (index_finger.x - thumb.x)**2 + 
                (index_finger.y - thumb.y)**2
            )
            # print(distance)
            
            if distance < 0.03:
                pyautogui.click()
                cv2.circle(img, (index_x, index_y), 15, (0, 255, 0), cv2.FILLED)
    
    cTime = time.time() 
    fps = 1 / (cTime - pTime) 
    pTime = cTime
    
    cv2.putText(img, f'FPS {int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Result Hand", img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        print('Программа завершила свою работу')
        break

cap.release()
cv2.destroyAllWindows()