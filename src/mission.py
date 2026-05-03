from ultralytics import YOLO
model = YOLO('yolov8n.pt') 
import pymurapi
import cv2
import numpy as np
import time

model = YOLO('yolov8n.pt') 
auv = pymurapi.mur_init()

class Mission:
    def __init__(self):
        # Настройки из диалога
        self.speed = 45
        self.depth = 2.5
        self.side_power = 80
        self.side_sign = 1 
        
        self.target_yaw = auv.get_yaw()
        self.shot_targets = []
        self.mission_start_time = time.time()
        
        self.KP_YAW = 0.6
        self.KD_YAW = 0.3   
        self.prev_yaw_err = 0
        
        self.KP_DEPTH = 40.0
        self.KP_TRACKING = 0.45 
        self.ignore_classes = [9, 71]

    def setup(self):
        print("\n=== НАСТРОЙКА МИССИИ ===")
        try:
            d = input(f"Глубина [{self.depth}]: ")
            if d: self.depth = float(d)
            s = input(f"Скорость [{self.speed}]: ")
            if s: self.speed = int(s)
        except: pass
        self.target_yaw = auv.get_yaw()
        self.mission_start_time = time.time()

    def drive(self, forward, yaw_err, side=0):
        if yaw_err > 180: yaw_err -= 360
        if yaw_err < -180: yaw_err += 360
        
        # Стабилизация курса (PD)
        d_yaw = yaw_err - self.prev_yaw_err
        self.prev_yaw_err = yaw_err
        y_out = (yaw_err * self.KP_YAW) + (d_yaw * self.KD_YAW)
        
        d_err = self.depth - auv.get_depth()
        d_out = d_err * self.KP_DEPTH
        
        auv.set_motor_power(0, int(np.clip(forward + y_out, -100, 100)))
        auv.set_motor_power(1, int(np.clip(forward - y_out, -100, 100)))
        auv.set_motor_power(2, int(np.clip(-d_out, -100, 100)))
        auv.set_motor_power(3, int(np.clip(-d_out, -100, 100)))
        auv.set_motor_power(4, int(np.clip(side, -100, 100)))

    def wall_maneuver(self):
        """ Разворот с обязательной паузой на стабилизацию """
        direction = "ВПРАВО" if self.side_sign > 0 else "ВЛЕВО"
        print(f"!!! СТЕНА: Ухожу {direction} !!!")
        
        t_end = time.time() + 2.0
        while time.time() < t_end:
            self.drive(-30, 0)
            time.sleep(0.02)
            
        t_end = time.time() + 4.0
        while time.time() < t_end:
            y_err = self.target_yaw - auv.get_yaw()
            self.drive(0, y_err, side=self.side_power * self.side_sign) 
            time.sleep(0.02)
            
        self.target_yaw = (self.target_yaw + 180) % 360
        if self.target_yaw > 180: self.target_yaw -= 360
        
        print(">>> СТАБИЛИЗАЦИЯ ПОСЛЕ РАЗВОРОТА (2.5 сек)...")
        t_end = time.time() + 2.5
        while time.time() < t_end:
            y_err = self.target_yaw - auv.get_yaw()
            # Стоим на месте, держим курс
            self.drive(0, y_err) 
            time.sleep(0.02)
            
        self.side_sign *= -1
        self.mission_start_time = time.time()

    def process_frame(self, img):
        # Детекция YOLO
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, conf=0.25, verbose=False)
        
        annotated_frame = img.copy()
        found_target = None
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in self.ignore_classes: continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Рисуем рамку (твоя логика один в один)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                if found_target is None:
                    error_x = cx - img.shape[1] // 2
                    area = (x2 - x1) * (y2 - y1)
                    found_target = (error_x, area, (cx, cy))
                    
        return annotated_frame, found_target

   def run(self):
        self.setup()
        target_last_frame = False
        
        while True:
            frame = auv.get_image_front()
            if frame is None: continue

            output_img, target = self.process_frame(frame)

            if target:
                if not target_last_frame:
                    print(">>> ОБЪЕКТ ОБНАРУЖЕН! Перехожу к наведению.")
                
                target_last_frame = True # Фиксируем наличие цели
                error_x, area, coords = target
                
                print(f"ЦЕЛЬ ВИЖУ: ошибка {error_x}, площадь {area}")
                fwd = 0 if abs(error_x) > 40 else 15
                self.drive(fwd, error_x * self.KP_TRACKING)
                
                if abs(error_x) < 25 and area > 2500:
                    if not any(abs(coords[0]-tx)<70 and abs(coords[1]-ty)<70 for tx, ty in self.shot_targets):
                        print(">>> ОГОНЬ (X2) <<<")
                        auv.shoot(); time.sleep(0.4); auv.shoot()
                        self.shot_targets.append(coords)
                        time.sleep(1.2)
                        self.target_yaw = auv.get_yaw()
            else:
                if target_last_frame:
                    print("!!! ОБЪЕКТ ПОТЕРЯН ИЗ ВИДУ !!! Возврат в режим патрулирования.")
                
                target_last_frame = False 
                # Патрулирование
                p, r = auv.get_pitch(), auv.get_roll()
                if time.time() - self.mission_start_time > 4.5 and abs(p) < 0.02 and abs(r) < 0.02:
                    self.wall_maneuver()
                else:
                    y_err = self.target_yaw - auv.get_yaw()
                    self.drive(self.speed, y_err)

            cv2.imshow("Vision & Control", output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                cv2.destroyAllWindows()
if __name__ == "__main__":
    try: Mission().run()
    except KeyboardInterrupt:
        for i in range(5): auv.set_motor_power(i, 0)
