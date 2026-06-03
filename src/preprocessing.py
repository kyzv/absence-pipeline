import cv2
import numpy as np


def preprocess(image_path):
   image = load_image(image_path)
   gray = to_grayscale(image)
   normalized = normalize_lighting(gray)
   binary = binarize(normalized)
   clean = deskew(binary)
   return clean


def load_image(image_path):
   image = cv2.imread(image_path)
   if image is None:
      raise FileNotFoundError(f"No image found at path: {image_path}")
   return image


def to_grayscale(image):
   return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_lighting(gray_image):
    return cv2.equalizeHist(gray_image)
 
 
def binarize (gray_image):
   _, binary = cv2.threshold(
      gray_image, 0, 255,
      cv2.THRESH_BINARY + cv2.THRESH_OTSU,
   )
   return binary


def deskew(binary_image):
   coords = np.column_stack(np.where(binary_image < 128))
   angle = cv2.minAreaRect(coords)[-1]
   print(f"  angle: {angle}")
   if angle < -45:
      angle = 90 + angle
   if abs(angle) > 10:
      return binary_image
   print(f"  angle: {angle}")
   (h, w) = binary_image.shape[:2]
   center = (w // 2, h // 2)
   M = cv2.getRotationMatrix2D(center, angle, 1)
   deskewed = cv2.warpAffine(
      binary_image, M, (w, h),
      flags=cv2.INTER_CUBIC,
      borderMode=cv2.BORDER_REPLICATE
   )
   return deskewed
   

result = preprocess("./data/raw/test/attendance sheet.jpg")
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()