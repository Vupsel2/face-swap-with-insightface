import cv2
import insightface

import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

#plt.rcParams['figure.dpi'] = 300
#plt.rcParams['savefig.dpi'] = 300
print('insightface', insightface.__version__)

app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

image_path = '6.jpg'
img = cv2.imread(image_path)
# Конвертируем из одного пространства цветов в другой 1 способом
plt.imshow(img[:,:,::-1])
plt.axis('off')


# распознаем лица и записываем данные о них в переменную faces
faces = app.get(img)
# картинку на базе оригинальной, но с отрисованными областями на месте распознания лиц
rimg = app.draw_on(img, faces)
# сохраняем картинкуdeprecations
cv2.imwrite("./t1_output.jpg", rimg)

img_r = cv2.imread('t1_output.jpg')
# Конвертируем из одного пространства цветов в другой 2 способом
img2 = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB) 
# отключаем отображение осей
plt.axis('off')
# отображаем график
plt.imshow(img2)

print(len(faces))
fig, axs = plt.subplots(1,len(faces), figsize = (4,1.5))

for i, face in enumerate(faces):
  bbox = face['bbox']
  bbox = [int(b) for b in bbox]
  print("iiii",i)
  print(bbox)
  axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2],::-1])
  axs[i].axis('off')


swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download = True)


source_face = faces[0]
bbox = source_face['bbox']
bbox = [int(b) for b in bbox]
plt.figure(figsize = (1,1))
plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2],::-1], aspect='equal')
plt.axis('off')

res = img.copy()
for face in faces:
  res = swapper.get(res, face, source_face)
  
plt.imshow(res[:,:,::-1])
plt.axis('off')
cv2.imwrite("./t1_output.jpg", res)


plt.show()


