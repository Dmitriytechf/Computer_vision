from mtcnn import MTCNN
import cv2

# Загружаем детектор
detector = MTCNN()

fixed_width = 1360
fixed_height = 800

img = cv2.imread('images/funyhobbit.jpg')
# img = cv2.imread('images/bilbo.jpg') # Вот эта картинка тяжелая
# img = cv2.imread('images/fellowship.jpg')

image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Надо в rgb для mtcnn

faces = detector.detect_faces(image_rgb) # Детекция лица

# Перебираем и отрисовываем
for face in faces:
    # Получаем координаты и рисуем квадрат
    x, y, w, h = face['box'] 
    cv2.rectangle(img,  (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    # Можно отрисовать ключевые точки на лице
    # if 'keypoints' in face:  
    #     for key, value in face['keypoints'].items():
    #         cv2.circle(img, (int(value[0]), int(value[1])), 2, (255, 255, 255), -1)


resized_img = cv2.resize(img, (fixed_width, fixed_height)) # Фиксированная длина изображения
cv2.imshow('Face in images', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
