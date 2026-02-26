import cv2
from ultralytics import YOLO

class ObjectTracker:
    def __init__(self, model_path="best.pt", source=0, conf=0.7):
        """ตั้งค่าเริ่มต้นโหลดโมเดลและเตรียมกล้อง"""
        self.cap = cv2.VideoCapture(source)
        self.model = YOLO(model_path)
        self.conf_threshold = conf
        self.window_name = "Smart Detection System"

    def process_frame(self, frame):
        """ประมวลผลภาพ: ตรวจจับวัตถุและวาดจุดกึ่งกลาง"""
        # สั่งให้โมเดลทำ Tracking
        results = self.model.track(frame, persist=True, device="cpu", conf=self.conf_threshold, verbose=False)
        
        # ดึงภาพที่ YOLO วาดกรอบมาให้แล้ว
        annotated_frame = results[0].plot()

        # ตรวจสอบว่ามีวัตถุที่ถูก Track หรือไม่
        boxes = results[0].boxes
        if boxes.id is not None:
            # ดึงข้อมูลพิกัด (center_x, center_y, w, h)
            centers = boxes.xywh.cpu().numpy()
            
            for box in centers:
                cx, cy = int(box[0]), int(box[1])
                # วาดจุดสีแดง (Red Dot) ตรงกลาง
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        return annotated_frame

    def start(self):
        """เริ่มลูปการทำงานหลัก"""
        print(f"System Ready. Window: '{self.window_name}' - Press 'q' to exit.")
        
        try:
            while self.cap.isOpened():
                success, img = self.cap.read()
                if not success:
                    print("Camera error or stream ended.")
                    break

                # ส่งภาพไปประมวลผล
                output_img = self.process_frame(img)

                # แสดงผล
                cv2.imshow(self.window_name, output_img)

                # เช็คการกดปุ่มออก
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        """คืนทรัพยากรระบบ"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete.")

if __name__ == "__main__":
    # เรียกใช้งานผ่าน Class
    tracker = ObjectTracker(model_path="best.pt", conf=0.7)
    tracker.start()