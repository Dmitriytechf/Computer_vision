import cv2
import time
import numpy as np
import HandTrackingModul as htm
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL


# Задаем ширину и высоту камеры
wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0) # вебка
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=70) # Уведичил порог детекции объекта

# Инициализация Pycaw для управления громкостью
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Получаем диапазон громкости
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        print('Что-то пошло не так')
        break
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])
        # Координаты большого и указательного пальцев
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2) // 2, (y1+y2) // 2
        
        # Рисуем круги и линию между пальцами
        cv2.circle(img, (x1, y1), 15, (226, 7, 250), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (226, 7, 250), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (226, 7, 250), 3)
        # cv2.circle(img, (cx, cy), 15, (226, 7, 250), cv2.FILLED)
        
        # Вычисляем длину между пальцами
        length = math.hypot(x2-x1, y2-y1)
        
        vol = np.interp(length, [50, 290], [minVol, maxVol])
        volBar = np.interp(length, [50, 280], [400, 150])
        volPer = np.interp(length, [50, 280], [0, 100])
        print(int(length), vol)
        
        volume.SetMasterVolumeLevel(vol, None)
        
        if length<50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    
    cv2.rectangle(img, (50, 150), (85, 400), (2, 115, 4), 5)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)
    
    cTime = time.time() # Текущее время
    fps = 1 / (cTime - pTime) # FPS = 1 / время между кадрами
    pTime = cTime  # Запоминаем текущее время для следующего кадра
    
    cv2.putText(img, f'FPS {int(fps)}', (40, 122), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 3)
    
    cv2.imshow('Img', img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()