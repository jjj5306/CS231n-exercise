from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="coco8.yaml", epochs=3)

# Evaluate the model's performance on the validation set
results = model.val(plots=True)

# Perform object detection on an image using the model
results = model("https://ultralytics.com/images/bus.jpg")

# Save the detection result
results[0].save("detection_result.jpg")

# Export the model to ONNX format
success = model.export(format="onnx")

print("Validation results plots are saved in the 'runs/detect/val' directory")
print("Detection result saved as 'detection_result.jpg'")