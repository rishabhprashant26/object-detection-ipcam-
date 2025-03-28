import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (nano version for speed)
model = YOLO('yolov8n.pt')

# Set the IP camera URL (change this to your IP address)
ip_camera_url = "http://192.168.31.133:8080/video"  # Change to your IP camera URL

# Open the IP camera feed
cap = cv2.VideoCapture(ip_camera_url)

if not cap.isOpened():
    print("Error: Could not open IP camera feed.")
    exit()

# Start real-time object detection
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform detection on the frame
    results = model(frame)

    # Loop through detection results properly
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for i in range(len(boxes)):
            confidence = confidences[i].item()
            class_id = int(class_ids[i].item())
            detected_object = model.names[class_id]

            # Draw bounding box and label
            x1, y1, x2, y2 = boxes[i].tolist()
            label = f"{detected_object} {confidence:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow('YOLOv8 IP Camera Object Detection', frame)

    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
