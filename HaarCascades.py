import cv2


img = cv2.imread('images/obi_eni.jpg')
# img = cv2.imread('images/luke.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fixed_width = 1280
fixed_height = 800

# Загрузка предобученного кода с каскадом Хаара для лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_detect = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,  # Уменьшение масштаба на 10% на каждом шаге
    minNeighbors=5,    # Минимальное количество соседей для удержания прямоугольника
    minSize=(30, 30)   # Минимальный размер объекта
)

for (x, y, w, h) in face_detect:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

resized_img = cv2.resize(img, (fixed_width, fixed_height))
cv2.imshow('Haar Detect', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
