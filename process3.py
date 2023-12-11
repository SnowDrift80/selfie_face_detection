import dlib
import cv2
import numpy as np
import imutils
import uuid

class process_photo:
    def align_and_resize_face(image_path, target_resolution, zoomfactor=1.2):
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

            # Expand the bounding box to zoom out
            top = int(max(0, face.top() - zoom_out_factor * (face.bottom() - face.top())))
            bottom = int(min(rotated_face.shape[0], face.bottom() + zoom_out_factor * (face.bottom() - face.top())))
            left = int(max(0, face.left() - zoom_out_factor * (face.right() - face.left())))
            right = int(min(rotated_face.shape[1], face.right() + zoom_out_factor * (face.right() - face.left())))

            # Crop the expanded face
            aligned_face = rotated_face[top:bottom, left:right]

            # Resize the aligned face to the target resolution with supersampling
            aligned_face = imutils.resize(aligned_face, width=target_resolution, inter=cv2.INTER_LANCZOS4)

            # Apply a different sharpening method to the supersampled image
            aligned_face = cv2.detailEnhance(aligned_face, sigma_s=10, sigma_r=0.15)

            # Apply bilateral filter for color correction and artifact reduction
            aligned_face = cv2.bilateralFilter(aligned_face, d=9, sigmaColor=75, sigmaSpace=75)

            # Save the aligned and color-corrected face
            filename = f'./results/{uuid.uuid4()}.jpg'
            cv2.imwrite(filename, aligned_face)

