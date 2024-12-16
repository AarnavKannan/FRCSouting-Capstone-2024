import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import euclidean
from shapely.geometry import box, Point, Polygon
from ultralytics import YOLO
import numpy as np

# Function to preprocess video frames
def preprocess_frame(frame, img_size=640):
    frame_resized = cv2.resize(frame, (img_size, img_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_tensor = torch.tensor(frame_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return frame_tensor

# Function to parse detections from model output
def parse_detections(detections, conf_threshold=0.05):
    """
    Parse detections from model output.
    """
    boxes, scores, class_ids = [], [], []

    # Loop through each detection result (detections is a list of Results)
    for result in detections:
        # The result will have a .boxes attribute for bounding boxes and .scores for confidence scores
        boxes_tensor = result.boxes.xywh  # Assuming the bounding boxes are stored in this format
        scores_tensor = result.boxes.conf  # Confidence scores
        class_ids_tensor = result.boxes.cls  # Class IDs

        for i in range(len(boxes_tensor)):  # Iterate through each detection in the result
            detected_box = boxes_tensor[i].tolist()  # Get the bounding box [x_center, y_center, w, h]
            confidence = scores_tensor[i].item()  # Get the confidence score
            class_id = int(class_ids_tensor[i].item())  # Get the class ID

            if confidence > conf_threshold:  # Only consider detections with high confidence
                x1, y1, w, h = detected_box
                # x2, y2 = x1 + w, y1 + h  # Convert to [x1, y1, x2, y2]
                boxes.append([x1, y1, w, h])
                scores.append(confidence)
                class_ids.append(class_id)

    return boxes, scores, class_ids

# Function to log proximity interactions
def log_proximity(tracked_objects):
    for i, obj1 in enumerate(tracked_objects):
        for j, obj2 in enumerate(tracked_objects):
            if i < j:
                distance = euclidean(obj1["centroid"], obj2["centroid"])
                if distance < proximity_threshold:
                    file = open("logs.txt", "a")
                    file.write(f"Objects {obj1['id']} and {obj2['id']} are close. Distance: {distance}")

# Function to log collisions
def log_collisions(tracked_objects):
    for i, obj1 in enumerate(tracked_objects):
        for j, obj2 in enumerate(tracked_objects):
            if i < j:
                iou = calculate_iou(obj1["bbox"], obj2["bbox"])
                if iou > 0.5:
                    file = open("logs.txt", "a")
                    file.write(f"Objects {obj1['id']} and {obj2['id']} are colliding. IoU: {iou}")

# Function to calculate IoU
def calculate_iou(bbox1, bbox2):
    box1 = box(*bbox1)
    box2 = box(*bbox2)
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union

# Scale zones to resized dimensions
def scale_zone(zone, original_width, original_height, resized_width, resized_height):
    scaled_points = [
        (x * resized_width / original_width, y * resized_height / original_height)
        for x, y in zone
    ]
    return Polygon(scaled_points)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLO("C:\\Users\\akannan1\\Projects\\FinalCodes\\model_results2\\content\\runs\\detect\\train\\weights\\best.pt", verbose=False).to(device)
new_dataset_path = "C:\\Users\\akannan1\\Projects\\FinalCodes\\FRC Automatic Scoutingv19\\data.yaml"
results = model.val(data=new_dataset_path)

#Display validation results
print('-----Results-----')
print(results)

pytorch_model = model.model
pytorch_model.eval()

# Initialize DeepSORT tracker
tracker = DeepSort(
    max_age=30,  # Number of frames to retain a track without updates
    nn_budget=100,  # Budget for appearance embeddings
    max_iou_distance=0.7  # Maximum IoU distance for matching
)

# Define proximity threshold for interaction detection
proximity_threshold = 10  # Distance in pixels

#Create logs file
file = open("logs.txt", "x")

# Define class names here
class_names = ["BlueAmp", "BlueSpkr", "RedAmp", "RedSpkr", "note", "robot"]

# Open video and initialize writer
video_path = "C:\\Users\\akannan1\\Projects\\FinalCodes\\inputVideo.mp4"
cap = cv2.VideoCapture(video_path)

#Get video properties
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
input_fps = int(cap.get(cv2.CAP_PROP_FPS))

#Scaling debugging
# print(f"Original frame dimensions: {frame_width}x{frame_height}")
# print(f"Resized frame dimensions: 640x640")

# Define zones
resized_width, resized_height = 640, 640

# Define zones in original dimensions
blue_wing_original = [(411, 430), (807, 433), (695, 644), (262, 621), (246, 537)]
red_wing_original = [(1123, 434), (1508, 433), (1643, 541), (1622, 637), (1187, 645)]
blue_wing = scale_zone(blue_wing_original, frame_width, frame_height, resized_width, resized_height)
red_wing = scale_zone(red_wing_original, frame_width, frame_height, resized_width, resized_height)

# Set output FPS
output_fps = 15

# Determine frame skipping interval
frame_skip = max(1, input_fps // output_fps)

out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (frame_width, frame_height))

#Track the frame index so we can skip frames
frame_index = 0

while cap.isOpened():
    # Read the current frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    # Skip frames to achieve the desired output FPS
    for _ in range(frame_skip - 1):
        if not cap.grab():
            break  # Exit if no frame is grabbed

    ret, frame = cap.retrieve()
    if not ret:
        break

    # Update which frame is currently being inferenced and display it
    print(f"Processing frame {frame_index + 1} / {total_frames}")
    frame_index += frame_skip

    # Preprocess frame for the model
    input_tensor = preprocess_frame(frame)

    with torch.no_grad():
        detections = model(input_tensor)
        # print(f"Type of detections: {type(detections)}")
        # print(f"Content of detections: {detections}")
        # print(f"Length of detections: {len(detections)}")
        # for detection in detections:
        #     print(f"{type(detection)}")

    # Parse detections (modify based on your model's output)
    boxes, scores, class_ids = parse_detections(detections)

    # Format detections for DeepSORT
    detections_for_tracking = []
    for detected_box, score, class_id in zip(boxes, scores, class_ids):
        detection = [detected_box, float(score), float(class_id)]
        detections_for_tracking.append(detection)

    # Print detections for debugging
    # print(detections_for_tracking)
    # print(f"Boxes: {type(boxes)} with shape {np.shape(boxes)}")
    # print(f"Scores: {type(scores)} with shape {np.shape(scores)}")
    # print("Formatted Detections:")
    # for detection in detections_for_tracking:
    #     print(detection)

    # Update DeepSORT tracker
    tracked_objects = tracker.update_tracks(detections_for_tracking, frame=frame)

    # Organize tracked data
    tracked_objects_data = []
    for obj in tracked_objects:
        bbox = obj.to_tlbr()  # Get bounding box [x1, y1, x2, y2]
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        tracked_objects_data.append({"id": obj.track_id, "bbox": bbox, "centroid": centroid})

    # Log proximity interactions
    log_proximity(tracked_objects_data)

    # Log collisions
    log_collisions(tracked_objects_data)

    # Check zone interactions
    for obj in tracked_objects_data:
        if blue_wing.contains(Point(obj["centroid"])):
            file = open("logs.txt", "a")
            file.write(f"Object {obj['id']} is in the Blue Wing.")
        if red_wing.contains(Point(obj["centroid"])):
            file = open("logs.txt", "a")
            file.write(f"Object {obj['id']} is in the Red Wing.")

    # Define the region where annotations are allowed
    allowed_region = Polygon([(1, 255), (1919, 255), (1919, 675), (1, 675)])

    # Annotate frame with tracking results
    for obj in tracked_objects_data:
        #print(obj)
        x1, y1, x2, y2 = map(int, obj["bbox"])
        obj_id = obj["id"]

        # Scale bounding box coordinates to match original frame dimensions
        x1 = int(x1 * frame_width / 640)
        y1 = int(y1 * frame_height / 640)
        x2 = int(x2 * frame_width / 640)
        y2 = int(y2 * frame_height / 640)

        # Check if centroid is within the allowed region
        centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
        if allowed_region.contains(Point(centroid)):
            # Draw bounding box and text only in the allowed region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Id:{obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write annotated frame to output
    out.write(frame)

cap.release()
out.release()