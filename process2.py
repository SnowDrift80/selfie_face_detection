import dlib
import cv2
import numpy as np
import imutils

def align_and_resize_face(image_path, target_resolution):
    # Load the pre-trained facial landmarks predictor from dlib
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Load an image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use dlib to detect faces
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray)

    for face in faces:
        # Use dlib to find facial landmarks
        landmarks = predictor(gray, face)

        # Extract facial landmarks (you may need to adjust these indices based on your needs)
        left_eye = landmarks.part(36).x, landmarks.part(36).y
        right_eye = landmarks.part(45).x, landmarks.part(45).y

        # Calculate the angle for alignment
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Rotate and align the face
        rotated_face = imutils.rotate(image, angle)

        # Crop the aligned face
        # (you may need to adjust the coordinates based on your needs)
        aligned_face = rotated_face[face.top():face.bottom(), face.left():face.right()]

        # Resize the aligned face to the target resolution with supersampling
        aligned_face = imutils.resize(aligned_face, width=target_resolution, inter=cv2.INTER_LANCZOS4)

        # Apply sharpening to the supersampled image
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        aligned_face = cv2.filter2D(aligned_face, -1, sharpening_kernel)

        # Save the aligned and sharpened face
        cv2.imwrite("aligned_and_sharpened_face3.jpg", aligned_face)

# Example usage with a target resolution of 300 pixels
align_and_resize_face("000009.jpg", target_resolution=300)
