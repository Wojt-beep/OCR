import cv2
from PIL import Image
import pytesseract
import numpy as np

img = cv2.imread('/Users/wojtek/Desktop/OCR/Znak.webp')

blurred = cv2.GaussianBlur(img, (7, 7), 0)

gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

ret, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

edges = cv2.Canny(gray, 100, 200)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=100,
    param2=30,
    minRadius=20,
    maxRadius=150
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    roi = img[y - r:y + r, x - r:x + r]

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)

    text = pytesseract.image_to_string(
        Image.fromarray(roi_thresh),
        config='--psm 8 -c tessedit_char_whitelist=0123456789'
    )

cv2.imshow("Roi", roi_thresh)
print("Odczytany tekst:", text.strip())
cv2.waitKey(0)
cv2.destroyAllWindows()
