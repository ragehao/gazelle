import cv2
import numpy as np
from PIL import Image
import torch
from gazelle.model import get_gazelle_model
from datetime import datetime

model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout')
model.load_gazelle_state_dict(torch.load("checkpoint/gazelle_dinov2_vitb14_inout.pt", weights_only=True))
model.eval()

device = "cuda" 
if torch.cuda.is_available():
    print("Using GPU for inference")
else:
    device = "cpu"
    print("Using CPU for inference")
    
model.to(device)

from gazelle.utils import visualize_heatmap
import threading
import time

# Capture video from camera using a background thread that always keeps latest frame
stream_url = "udp://10.42.0.103:8554?fifo_size=50000000&overrun_nonfatal=1"

class FrameGrabber(threading.Thread):
    """Background thread that continuously reads frames and stores the latest one."""
    def __init__(self, src):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                # small sleep to avoid tight loop if stream temporarily unavailable
                time.sleep(0.01)
                continue
            with self.lock:
                self.ret = True
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.running = False
        try:
            self.join(timeout=1.0)
        except RuntimeError:
            pass
        try:
            self.cap.release()
        except Exception:
            pass

grabber = None
try:
    grabber = FrameGrabber(stream_url)
except RuntimeError as e:
    print(e)
    exit()

grabber.start()
print("Press 'q' to quit.")

while True:
    ret, frame = grabber.read()
    if not ret:
        # no valid frame yet
        time.sleep(0.01)
        continue

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    image = Image.fromarray(frame_rgb)

    input_data = {
        "images": transform(image).unsqueeze(dim=0).to(device),    # tensor of shape [1, 3, 448, 448]
        "bboxes": [[None]]            # list of lists of bbox tuples
    }

    with torch.no_grad():
        output = model(input_data)
    predicted_heatmap = output["heatmap"][0][0]        # access prediction for first person in first image. Tensor of size [64, 64]
    predicted_inout = output["inout"][0][0]            # in/out of frame score (1 = in frame)

    # Visualize heatmap
    viz = visualize_heatmap(image, predicted_heatmap)
    # Convert PIL to numpy array
    viz_np = np.array(viz)
    # Convert RGB to BGR for OpenCV
    viz_bgr = cv2.cvtColor(viz_np, cv2.COLOR_RGB2BGR)

    # Print timestamp and prediction info
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{ts}  Predicted in-out score: {predicted_inout.item():.4f}")

    # Overlay timestamp on the frame (top-left)
    cv2.putText(viz_bgr, ts, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Display the result
    cv2.imshow('Gaze Prediction', viz_bgr)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if grabber is not None:
    grabber.stop()
cv2.destroyAllWindows()