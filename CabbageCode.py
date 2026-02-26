#ก่อน Run ให้ Activate venv ก่อน
# venv\Scripts\activate

#แล้วใช้คำสั่งนี้ในการรัน
#python CabbageCode.py

import cv2
from ultralytics import YOLO

class ObjectTracker:
    # ปรับค่า default conf จาก 0.35 เป็น 0.70 (หรือค่าที่คุณต้องการ)
    def __init__(self, model_path="best.pt", source=0, conf=0.7):
        """ตั้งค่าเริ่มต้นโหลดโมเดลและเตรียมกล้อง"""
        self.cap = cv2.VideoCapture(source)
        self.model = YOLO(model_path)
        self.conf_threshold = conf
        self.window_name = "Smart Detection System"
        
        # [เพิ่ม] ตัวแปรสำหรับนับจำนวนสะสม
        self.total_count = 0
        self.counted_ids = set() # ใช้เก็บ ID ที่เคยนับไปแล้ว
        
        # [เพิ่ม] ตัวแปรเทียบสเกล (ต้องลองปรับเลขนี้ดูครับ)
        # สมมุติที่ระยะ 20cm: 40 พิกเซล = 1 เซนติเมตร
        self.pixels_per_cm = 40.0 

    def process_frame(self, frame):
        """ประมวลผลภาพ: ตรวจจับวัตถุและวาดจุดกึ่งกลาง"""
        # สั่งให้โมเดลทำ Tracking
        results = self.model.track(frame, persist=True, device="cpu", conf=self.conf_threshold, verbose=False)
        
        # ดึงภาพที่ YOLO วาดกรอบมาให้แล้ว
        annotated_frame = results[0].plot()

        # ตรวจสอบว่ามีวัตถุที่ถูก Track หรือไม่
        boxes = results[0].boxes
        
        # ==========================================
        # [แก้ไข] ส่วนแสดงผลนับจำนวนสะสม (Total)
        # ==========================================
        # แสดงยอดสะสมที่มุมจอแทนยอดปัจจุบัน
        cv2.putText(annotated_frame, f"Total Count: {self.total_count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        # ==========================================

        # ต้องเช็คทั้ง boxes และ boxes.id ว่าไม่เป็น None
        if boxes is not None and boxes.id is not None:
            # ดึงข้อมูลพิกัด (center_x, center_y, width, height)
            # xywh = แกนx, แกนy, ความกว้าง, ความสูง
            box_data = boxes.xywh.cpu().numpy()
            track_ids = boxes.id.int().cpu().tolist()
            
            # วนลูปเช็คกะหล่ำแต่ละหัวที่เจอในภาพ
            for box, track_id in zip(box_data, track_ids):
                cx, cy, w, h = box
                
                # --- 1. ตรรกะการนับสะสม ---
                # ถ้า ID นี้ยังไม่เคยถูกนับ ให้บวกเพิ่ม
                if track_id not in self.counted_ids:
                    self.total_count += 1
                    self.counted_ids.add(track_id)
                
                # --- 2. การคำนวณขนาด (cm) ---
                width_cm = w / self.pixels_per_cm
                height_cm = h / self.pixels_per_cm
                
                # วาดจุดสีแดงตรงกลาง
                cv2.circle(annotated_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                
                # เขียนบอกขนาดที่ตัวกะหล่ำ
                size_text = f"{width_cm:.1f}x{height_cm:.1f} cm"
                cv2.putText(annotated_frame, size_text, (int(cx) - 50, int(cy) - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return annotated_frame

    def start(self):
        """เริ่มลูปการทำงานหลัก"""
        print(f"System Ready. Window: '{self.window_name}'")
        print(f"Confidence Level: {self.conf_threshold}")
        print("Press 'q' to exit.")
        
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
    # เรียกใช้งานผ่าน Class พร้อมระบุค่า conf ที่ต้องการ (เช่น 0.7 หรือ 0.8)
    tracker = ObjectTracker(model_path="best.pt", conf=0.7)
    tracker.start()