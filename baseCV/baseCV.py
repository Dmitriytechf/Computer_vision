import cv2

# OpenCV (Open Source Computer Vision Library). Содержит 2500+ алгоритмов
# pip install opencv-python


img = cv2.imread('images/carsign1.jpg')
if img is None:
    print("Не удалось загрузить изображение. Проверь путь!")
    exit()

resize_img = cv2.resize(img, (1100, 700))
img_gray = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)


cv2.imshow('Window', img_gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
