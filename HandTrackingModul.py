import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=50, trackCon=50): 
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon  
        self.trackCon = trackCon
    
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon / 100.0, 
            min_tracking_confidence=self.trackCon / 100.0
        )
        self.mpDraw = mp.solutions.drawing_utils
        
        # Настройка стиля отрисовки
        self.hand_landmark_style = self.mpDraw.DrawingSpec(
            color=(0, 0, 255),  
            thickness=-1,       
            circle_radius=6   
        )

        self.hand_connection_style = self.mpDraw.DrawingSpec(
            color=(0, 255, 0),  
            thickness=5         
        )
        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, 
                        handLms, 
                        self.mpHands.HAND_CONNECTIONS,
                        self.hand_landmark_style,
                        self.hand_connection_style
                        )
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = [] # Создаём пустой список для хранения координат точек
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] 
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape # Получаем высоту, ширину и каналы изображения
                cx, cy = int(lm.x * w), int(lm.y * h) # Перевод в абсолютные координаты
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return
    
    detector = handDetector(detectionCon=50, trackCon=50)
    
    while True:
        success, img = cap.read()
        if not success:
            print("Не удалось получить кадр")
            break
            
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Hand Tracking", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()