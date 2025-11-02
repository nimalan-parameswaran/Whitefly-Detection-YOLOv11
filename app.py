import os
import cv2
import gradio as gr
from ultralytics import YOLO

# Load ONNX model
model = YOLO("best.onnx")

# Parameters (same as your notebook)
imgsz = 1280
conf = 0.2
max_det = 5000
shrink_ratio = 0.6  # smaller bounding boxes

def detect_whitefly(image):
    # Convert to OpenCV format
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save temporarily (YOLO.predict() needs a file path)
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    # Run prediction
    results = model.predict(
        source=temp_path,
        imgsz=imgsz,
        conf=conf,
        max_det=max_det,
        save=False
    )

    # Get first result
    r = results[0]
    img = img_rgb.copy()

    # Draw smaller bounding boxes with index numbers
    for idx, box in enumerate(r.boxes.xyxy, start=1):
        x1, y1, x2, y2 = map(int, box)

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w // 2
        cy = y1 + h // 2
        w_new = int(w * shrink_ratio)
        h_new = int(h * shrink_ratio)
        x1_new = cx - w_new // 2
        y1_new = cy - h_new // 2
        x2_new = cx + w_new // 2
        y2_new = cy + h_new // 2

        # Draw rectangle
        cv2.rectangle(img, (x1_new, y1_new), (x2_new, y2_new), (255, 0, 0), 2)
        # Draw index number above the box
        # cv2.putText(img, str(idx), (x1_new, y1_new - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Count detections
    count = len(r.boxes)

    # Draw total count
    cv2.putText(img, f"Total Whiteflies: {count}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

    return img, f"Detected Whiteflies: {count}"

# Gradio interface
demo = gr.Interface(
    fn=detect_whitefly,
    inputs=gr.Image(type="numpy", label="Upload Leaf Image"),
    outputs=[gr.Image(label="Detection Result"), gr.Textbox(label="Count")],
    title="Whitefly Detection",
    description="Upload any leaf image to detect and count whiteflies with indexed bounding boxes."
)

if __name__ == "__main__":
    demo.launch()
