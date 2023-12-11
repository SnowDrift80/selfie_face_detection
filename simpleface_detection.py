import cv2


# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image
image = cv2.imread('faces.jpg')

#Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    new_h = int(h * 1.1)
    new_y = int(y*.55)
    cv2.rectangle(image, (x, new_y), (x+w, y+new_h), (255, 0, 0), 2)
    
    # Extract the region of interest (face)
    face_roi = image[new_y:y + new_h, x:x + w]
    
    # Save each face as a sepearte JPG file
    cv2.imwrite(f"face_{x}_{y}.jpg", face_roi)

# Display the result
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()