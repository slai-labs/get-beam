import shutil
import os
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def process_images(**inputs):
    # For static images:
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        # Read each file from the Persistent Volume
        for file in os.listdir("/volumes/unprocessed_images"):
            image = cv2.imread(str(file))
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print("RESULTS:", results)
            # Draw face detections of each face.
            if not results.detections:
                print("No faces found, skipping...")
                continue

            annotated_image = image.copy()
            print("ANNOTATED_IMAGE", annotated_image)

            for detection in results.detections:
                print("Nose tip:")
                print(
                    mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
                    )
                )
                mp_drawing.draw_detection(annotated_image, detection)
            # Outputs will be saved to the Output file mounted in app.py
            # These files will appear in the web dashboard if you've deployed your app
            # When working locally, you can download the files by running this command in your terminal:
            # > download output.png
            cv2.imwrite("output.png", annotated_image)


if __name__ == "__main__":
    # ** For your reference **
    # All the files in your local working directory are in /workspace/
    # All the files in your Persistent Volume are in /volumes/

    # Copies files from your local to a Persistent Volume
    for file in os.listdir("/workspace/images"):
        print("Copying file", file)
        shutil.copyfile(
            f"/workspace/images/{file}",
            f"/volumes/unprocessed_images/{file}",
        )

    process_images()
