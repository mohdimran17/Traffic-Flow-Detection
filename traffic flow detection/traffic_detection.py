import cv2
from ultralytics import YOLO
import pandas as pd

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load the input video to process the data

cap = cv2.VideoCapture("traffic_video.mp4")
frame_id = 0
vehicle_data = []

# Define vehicle classes from COCO dataset

VEHICLE_CLASSES = [2, 3, 5, 7]    # count of the vehicles - Car, Motorcycle, Bus, Truck

def count_vehicles_and_annotate(results):
    count = 0
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls in VEHICLE_CLASSES:
                count += 1
    return count



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    # Run YOLO every 10th frame

    if frame_id % 10 == 0:
        results = model(frame)
        count = count_vehicles_and_annotate(results)
        
        # Store count
        vehicle_data.append({'frame_id': frame_id, 'vehicle_count': count})

        # Draw detections on frame
        annotated_frame = results[0].plot()

        # Show count on frame
        cv2.putText(annotated_frame, f'Vehicles: {count}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Traffic Flow Detection', annotated_frame)

    frame_id += 1

    # Press 'q' to quit

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


# Save to CSV

df = pd.DataFrame(vehicle_data)
df.to_csv("traffic_counts.csv", index=False)
print("Vehicle count data saved to 'traffic_counts.csv'") 
