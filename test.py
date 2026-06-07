from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='fr', use_textline_orientation=True)
result = ocr.ocr('data/cropped/as1_header.jpg')
for line in result[0]:
   print(line)