import cv2
#Importimg the original image
image1=cv2.imread("C:\\Users\\Arjun\\Desktop\\20170409_165811-1-1170x878.jpg")
print(image1.shape)
#Resizing the original image
image=cv2.resize(image1,(0,0),None,0.5,0.5)
cv2.imshow("legend",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#loading the face_cascade for the face detection
face_cascade=cv2.CascadeClassifier("C:\\Users\\Arjun\\Desktop\\haarcascade_frontalcatface.xml")
#Creation of Gray_Scale image
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#Detecting the multiface
faces = face_cascade.detectMultiScale(
      gray_image,
      scaleFactor=1.002
  )

print(type(faces))
print(faces)
#Creating the rectangular boundries
for x,y,w,h in faces:
    image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    
#finding the number of faces
print("Found {0} faces!".format(len(faces)))
#resizing the image
resized=cv2.resize(image,(int(image.shape[1]/7),int(image.shape[0]/7)))
cv2.imshow("grey",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#heat map
heatmap_img = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
cv2.imshow("heatmap_img",heatmap_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Semantic segmentation
pip install tensorflow
pip install pip-upgrade-outdated
pip install pillow
pip install scikit-image
import pixellib
import tensorflow as tf
from pixellib.semantic import semantic_segmentation 
segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model('C:/Users/Arjun/Desktop/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5')  #  deeplabv3_xception_tf_dim_ordering_tf_kernels
segment_image.segmentAsPascalvoc('C:/Users/Arjun/Desktop/crowd.jpg', output_image_name = 'C:/Users/Arjun/Desktop/%s_image_new_deeplabv3.jpg')