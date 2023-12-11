import dlib
import cv2
import numpy as np
import imutils
import uuid
import os
import shutil

class ProcessPhoto:
    @staticmethod
    def align_and_resize_face(image_path, target_resolution, zoom_out_factor=0.3, saturation_factor=0.8, brightness_factor=0.7):
        # Load the pre-trained facial landmarks predictor from dlib
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Load an image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use dlib to detect faces
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        
        print(f"{len(faces)} faces detected! \n")

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
            aligned_face = cv2.detailEnhance(aligned_face, sigma_s=4, sigma_r=0.02)

            # Apply bilateral filter for color correction and artifact reduction
            aligned_face = cv2.bilateralFilter(aligned_face, d=9, sigmaColor=75, sigmaSpace=75)

            # Convert the image from BGR to HSV
            aligned_face_hsv = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2HSV)

            # Adjust the saturation
            aligned_face_hsv[:, :, 1] = np.clip(aligned_face_hsv[:, :, 1] * saturation_factor, 0, 255)

            # Adjust the brightness
            aligned_face_hsv[:, :, 2] = np.clip(aligned_face_hsv[:, :, 2] * brightness_factor, 0, 255)

            # Convert the image back to BGR
            aligned_face = cv2.cvtColor(aligned_face_hsv, cv2.COLOR_HSV2BGR)

            # Save the aligned, color-corrected, normalized saturation, and normalized brightness face
            filename = f'./results/{uuid.uuid4()}.jpg'
            cv2.imwrite(filename, aligned_face)
            
        # Move the processed file from the original location to the "./processed" directory
        processed_directory = "./processed"
        os.makedirs(processed_directory, exist_ok=True)
        processed_path = os.path.join(processed_directory, os.path.basename(image_path))
        shutil.move(image_path, processed_path)
