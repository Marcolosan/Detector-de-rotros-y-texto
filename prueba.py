# Instalación y reinicio

!sudo apt install tesseract-ocr
!pip install pytesseract # RT

!pip install Pillow==9.0.0 # RT

!pip install easyocr

!pip install "paddleocr>=2.0.1"

!pip install face_recognition

# Importar librerias

!tesseract p_h_8.jpeg - -l eng

# import the necessary packages
from pytesseract import Output
import pytesseract
import argparse
import cv2
import dlib
from google.colab.patches import cv2_imshow # para cv2.imshow(img)

# load example images
!npx degit JaidedAI/EasyOCR/examples -f

#!pip install paddlepaddle-gpu # Con GPU
!pip install paddlepaddle

!paddleocr --image_dir lecture01.jpeg --lang=en

!wget https://github.com/Halfish/lstm-ctc-ocr/raw/master/fonts/simfang.ttf

from google.colab import drive
drive.mount('/content/drive')

# Foto 1

foto= "/content/drive/MyDrive/semestre 5/Fotos Lecture/f1.jpeg"


from PIL import Image, ImageDraw
import face_recognition


**Detecta el rostro**

# DETECTA LOS ROSTROS


# Load the jpg file into a numpy array
family_photo = face_recognition.load_image_file(foto)

# Convert the group photo into a PIL Image
family_pil_image = Image.fromarray(family_photo)

# Find all the faces in the image 
fl = face_recognition.face_locations(family_photo)
draw = ImageDraw.Draw(family_pil_image)

from PIL import Image
import PIL
from PIL import ImageDraw
# Let us print the number of faces in the Photo
face_count = len(fl)
print("No. of Faces detected in this photo", face_count)

for i in range(face_count):
    # Print the location of each face in this image
    top, right, bottom, left = fl[i]
    print("Face,,", i, " Top, Left, , Bottom, Right..", top, left, bottom, right)
    # You can access the actual face itself like this:
    draw.rectangle(
            (left, top, right, bottom),
            outline=(255, 0, 0), width=3)


family_pil_image



tesseract-ocr


from google.colab.patches import cv2_imshow
imagen = (foto)
nivel_confianza = 0.
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to localize each area of text in the input image
img = cv2.imread(imagen)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_data(rgb, output_type=Output.DICT)

# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
	# extract the bounding box coordinates of the text region from
	# the current result
	x = results["left"][i]
	y = results["top"][i]
	w = results["width"][i]
	h = results["height"][i]
	# extract the OCR text itself along with the confidence of the
	# text localization
	text = results["text"][i]
	conf = int(results["conf"][i])

# filter out weak confidence text localizations
	if conf >= nivel_confianza:
		# display the confidence and text to our terminal
		print("Confidence: {}".format(conf))
		print("Text: {}".format(text))
		print("")
		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw a bounding box around the text along
		# with the text itself
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
			1.2, (0, 0, 255), 3)

# show the output image
#cv2.imshow("Image", img)
#cv2.waitKey(0)
#img= cv2.resize(img,(450, 450), interpolation=cv2.INTER_LINEAR) # La hace más peuqeña




cv2_imshow(img)

from PIL import Image
extractedInformation = pytesseract.image_to_string(Image.open(foto))
print(extractedInformation)

**Easy OCR**



# show an image
import PIL
from PIL import ImageDraw
im = PIL.Image.open(foto)
im

# Create a reader to do OCR.
# If you change to GPU instance, it will be faster. But CPU is enough.
# (by MENU > Runtime > Change runtime type > GPU, then redo from beginning )
import easyocr
reader = easyocr.Reader(['en'])

# Doing OCR. Get bounding boxes.
bounds = reader.readtext(foto)
bounds

# Draw bounding boxes
def draw_boxes(image, bounds, color='yellow', width=2):
    img1 = im.resize((1024, 1024), Image.BOX)
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0, p1, p2, p3 = bound[0]
        draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
    return image

draw_boxes(im, bounds)

**PaddleOCR**

from google.colab.patches import cv2_imshow
from paddleocr import PaddleOCR,draw_ocr
import cv2
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = foto
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)


# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='simfang.ttf')
rgb = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
#im_show = Image.fromarray(im_show)
#im_show.save('result.jpg')
cv2_imshow(rgb)
