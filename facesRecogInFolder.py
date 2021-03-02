# USAGE
# Run detect_all_images.py. It gets images from /images folder and saves them to blurred_faces folder as 'blurred.IMAGENAME'

import numpy as np
import cv2
import os

if not os.path.exists('images'):
	os.makedirs('images')
	print('[INFO] images Folder Created.')
if not os.path.exists('files_faces'):
	os.makedirs('files_faces')
	print('[INFO] files_faces Folder Created.')

# Можно подкрутить, что улучшить/ухудшить обнаружение
args = {
	"image":"",
	"model":"res10_300x300_ssd_iter_140000.caffemodel",
	"prototxt":"deploy.prototxt.txt",
	"confidence": 0.9
}

# Загрузка модели
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


print("[INFO] computing object detections...")
for root, dirs, files in os.walk('images'):
	for i in files:
		# Загрузка входного изображения + создание blob объекта для изображения
		# Изменение размера до 300х300 и нормализация
		args["image"] = i
		image = cv2.imread("images/" + args["image"])
		print("images/" + args["image"])
		result_image= image.copy()
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		# Передача Blob объекта
		net.setInput(blob)
		detections = net.forward()

		# Цикл обнаружения
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			# Фильтрация маловероятных обнаружений для преодоления порога достоверности
			if confidence > args["confidence"]:
				# Вычисление координат рамки
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")


				files_name = 'files.'+ args['image']
				cv2.imwrite('files_faces/'+ args['image'] , result_image)
				# Для просмотра сразу друг за другом
				#cv2.imshow("Output", result_image)
	print("Saving " + 'files.'+ args['image'] + " in blurred_faces folder")
		#Для просмотра каждого изображения
		#cv2.waitKey(0)


