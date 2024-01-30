import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-obb.pt')  # load an official model

# source = "D:\\20_Data\\[20211029] 화성 ITS 영상\\02번 21331540(내각).avi"
source = "D:\\20_Data\\[20211124] 공주 ITS 영상\\mkv 영상\\06. 옥룡교차로 [FHD].mkv"
cap = cv2.VideoCapture(source)

# # # Run inference on the source
# results = model(source, stream=True, show=True, device=0)  # generator of Results objects


# # Process results generator
# for result in results:
#     boxes = result.boxes  # Boxes object for bbox outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, device=0)

        # Visualize the results on the frame
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs

            print(boxes)
            print(masks)
            print(keypoints)
            print(probs)

        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()