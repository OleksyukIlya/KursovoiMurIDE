from ultralytics import YOLO
import pymurapi
import cv2
import numpy as np
import time

auv = pymurapi.mur_init()

model = YOLO('yolov8n.pt')

class Mission:
    def __init__(self):
        self.speed = 45
        self.depth = 2.5
        self.side_power = 80
        self.side_sign = 1 
        
        self.target_yaw = auv.get_yaw()
        self.target_depth = self.depth
        self.shot_targets = []
        self.mission_start_time = time.time()
        
        self.KP_YAW = 0.6
        self.KD_YAW = 0.3   
        self.prev_yaw_err = 0
        
        self.KP_DEPTH = 40.0
        
        self.CONF_THRESHOLD = 0.25
        self.ignore_classes = [9, 71]

    def setup(self):
        print("\n=== НАСТРОЙКА МИССИИ ===")
        try:
            d = input("Глубина [" + str(self.depth) + "]: ")
            if d: 
                self.depth = float(d)
                self.target_depth = self.depth
            s = input("Скорость [" + str(self.speed) + "]: ")
            if s: 
                self.speed = int(s)
        except: 
            pass
        self.target_yaw = auv.get_yaw()
        self.mission_start_time = time.time()
        print("Стартовая глубина: " + str(self.depth) + " м")
        print("Стартовый курс: " + str(self.target_yaw) + " град")

    def drive(self, forward, yaw_err, depth_correction=0, side=0):
        if yaw_err > 180: 
            yaw_err -= 360
        if yaw_err < -180: 
            yaw_err += 360
        
        d_yaw = yaw_err - self.prev_yaw_err
        self.prev_yaw_err = yaw_err
        y_out = (yaw_err * self.KP_YAW) + (d_yaw * self.KD_YAW)
        
        target_d = self.target_depth + depth_correction
        d_err = target_d - auv.get_depth()
        d_out = d_err * self.KP_DEPTH
        
        auv.set_motor_power(0, int(np.clip(forward + y_out, -100, 100)))
        auv.set_motor_power(1, int(np.clip(forward - y_out, -100, 100)))
        auv.set_motor_power(2, int(np.clip(-d_out, -100, 100)))
        auv.set_motor_power(3, int(np.clip(-d_out, -100, 100)))
        auv.set_motor_power(4, int(np.clip(side, -100, 100)))

    def wall_maneuver(self):
        if self.side_sign > 0:
            direction = "ВПРАВО"
        else:
            direction = "ВЛЕВО"
            
        print("!!! СТЕНА: НАЧАЛО МАНЕВРА !!!")
        print("  >> Отъезд назад и смещение " + direction)
        
        t_end = time.time() + 2.0
        while time.time() < t_end:
            self.drive(-30, 0, side=self.side_power * self.side_sign)
            time.sleep(0.02)
        
        print("  >> Остановка перед разворотом")
        for _ in range(10):
            self.drive(0, 0)
            time.sleep(0.05)
        
        start_yaw = auv.get_yaw()
        print("  >> Начальный курс: " + str(start_yaw) + " град")
        
        print("  >> Разворот на 180 градусов")
        
        target_angle = 180
        turned_degrees = 0
        last_yaw = start_yaw
        
        if self.side_sign > 0:
            turn_speed = -35
        else:
            turn_speed = 35
        
        turn_start_time = time.time()
        
        while turned_degrees < target_angle - 10:
            current_yaw = auv.get_yaw()
            
            delta = current_yaw - last_yaw
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            
            turned_degrees += abs(delta)
            last_yaw = current_yaw
            
            self.drive(0, turn_speed)
            time.sleep(0.05)
            
            if time.time() - turn_start_time > 12:
                print("  !! Таймаут разворота")
                break
        
        self.target_yaw = auv.get_yaw()
        print("  >> Конечный курс: " + str(self.target_yaw) + " град")
        
        print("  >> Стабилизация курса")
        t_end = time.time() + 2.0
        while time.time() < t_end:
            y_err = self.target_yaw - auv.get_yaw()
            if y_err > 180:
                y_err -= 360
            if y_err < -180:
                y_err += 360
            self.drive(0, y_err)
            time.sleep(0.02)
        
        self.side_sign = self.side_sign * -1
        self.mission_start_time = time.time()
        print("  >> Маневр завершен\n")

    def process_frame(self, img):
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, conf=self.CONF_THRESHOLD, verbose=False)
        
        annotated_frame = img.copy()
        all_targets = []
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls in self.ignore_classes: 
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                confidence = float(box.conf[0])
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)
                
                conf_text = str(round(confidence, 2))
                cv2.putText(annotated_frame, conf_text, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                error_x = cx - img.shape[1] // 2
                error_y = cy - img.shape[0] // 2
                all_targets.append((error_x, error_y, confidence, (cx, cy), (x1, y1, x2, y2)))
        
        found_target = all_targets[0] if all_targets else None
        return annotated_frame, found_target

    def aim_and_shoot(self, target_info, frame_width, frame_height):
        error_x, error_y, confidence, coords, bbox = target_info
        
        print("")
        print("=== НАЧАЛО ПРИЦЕЛИВАНИЯ ===")
        print("Цель на позиции: X=" + str(coords[0]) + ", Y=" + str(coords[1]))
        
        start_depth = auv.get_depth()
        self.target_depth = start_depth
        
        for _ in range(10):
            self.drive(0, 0)
            time.sleep(0.05)
        
        aim_start_time = time.time()
        aim_timeout = 15
        last_shot_coords = None
        last_frame = None
        
        while time.time() - aim_start_time < aim_timeout:
            frame = auv.get_image_front()
            if frame is None:
                continue
            
            annotated_frame, target = self.process_frame(frame)
            last_frame = annotated_frame
            
            if target is None:
                print("Цель потеряна! Прерывание прицеливания")
                cv2.imshow("Vision & Control", last_frame)
                cv2.waitKey(1)
                return False
            
            error_x, error_y, confidence, coords, bbox = target
            
            center_x = frame_width // 2
            center_y = frame_height // 2
            
            cv2.putText(annotated_frame, "AIMING MODE", (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Target offset X: " + str(error_x) + "  Y: " + str(error_y), 
                       (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            print("")
            print("--- КОРРЕКЦИЯ ---")
            print("Отклонение X: " + str(error_x) + " пикс, Y: " + str(error_y) + " пикс")
            print("Уверенность: " + str(round(confidence, 2)))
            
            if last_shot_coords is not None:
                if abs(coords[0] - last_shot_coords[0]) < 50 and abs(coords[1] - last_shot_coords[1]) < 50:
                    print("Цель уже поражена, пропускаем")
                    return False
            
            if abs(error_y) > 30:
                correction_strength = 0.003
                depth_change = -error_y * correction_strength
                depth_change = np.clip(depth_change, -0.5, 0.5)
                
                if depth_change > 0:
                    print("Коррекция глубины: ОПУСКАЮСЬ на " + str(abs(depth_change)) + " м")
                else:
                    print("Коррекция глубины: ПОДНИМАЮСЬ на " + str(abs(depth_change)) + " м")
                
                correction_time = time.time() + 0.8
                while time.time() < correction_time:
                    self.drive(0, 0, depth_correction=depth_change)
                    time.sleep(0.02)
            
            if abs(error_x) > 25:
                turn_correction = error_x * 0.35
                turn_correction = np.clip(turn_correction, -25, 25)
                print("Коррекция курса: ПОВОРОТ на " + str(turn_correction) + " град")
                
                correction_time = time.time() + 0.8
                while time.time() < correction_time:
                    self.drive(0, turn_correction)
                    time.sleep(0.02)
            
            if abs(error_x) < 35 and abs(error_y) < 35:
                print("")
                print("*** ЦЕЛЬ В ПРИЦЕЛЕ! ***")
                
                for _ in range(15):
                    self.drive(0, 0)
                    time.sleep(0.05)
                    final_frame, _ = self.process_frame(auv.get_image_front())
                    if final_frame is not None:
                        cv2.imshow("Vision & Control", final_frame)
                        cv2.waitKey(1)
                
                auv.shoot()
                time.sleep(0.3)
                auv.shoot()
                time.sleep(0.3)
                
                print(">>> ВЫСТРЕЛ ПРОИЗВЕДЕН! <<<")
                last_shot_coords = coords
                self.shot_targets.append(coords)
                
                time.sleep(1.0)
                return True
            
            cv2.imshow("Vision & Control", annotated_frame)
            cv2.waitKey(1)
            time.sleep(0.1)
        
        print("Таймаут прицеливания!")
        return False

    def run(self):
        self.setup()
        aiming_mode = False
        detection_counter = 0
        
        print("\n=== ЗАПУСК МИССИИ ===")
        print("q - выход")
        print("")
        
        while True:
            frame = auv.get_image_front()
            if frame is None: 
                continue
            
            frame_height, frame_width = frame.shape[:2]
            output_img, target = self.process_frame(frame)
            
            cv2.putText(output_img, "Depth: " + str(round(auv.get_depth(), 2)) + " | Target: " + str(round(self.target_depth, 2)), 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if aiming_mode:
                mode_text = "Mode: AIMING"
                mode_color = (0, 0, 255)
            else:
                mode_text = "Mode: PATROL"
                mode_color = (0, 255, 0)
            cv2.putText(output_img, mode_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
            
            if target and not aiming_mode:
                error_x, error_y, confidence, coords, bbox = target
                detection_counter += 1
                
                if detection_counter >= 3:
                    print("")
                    print(">>> ЦЕЛЬ ОБНАРУЖЕНА! <<<")
                    print("Координаты цели: X=" + str(coords[0]) + ", Y=" + str(coords[1]))
                    print("Уверенность: " + str(round(confidence, 2)))
                    
                    aiming_mode = True
                    
                    success = self.aim_and_shoot((error_x, error_y, confidence, coords, bbox), frame_width, frame_height)
                    
                    if success:
                        print("")
                        print("=== ЦЕЛЬ УСПЕШНО ПОРАЖЕНА! ===")
                    else:
                        print("")
                        print("=== НЕ УДАЛОСЬ ПОРАЗИТЬ ЦЕЛЬ ===")
                    
                    aiming_mode = False
                    detection_counter = 0
                    self.target_yaw = auv.get_yaw()
                    self.target_depth = self.depth
                    self.mission_start_time = time.time()
                    
                    print("Продолжаю патрулирование...")
                    print("")
                    
            elif target and aiming_mode:
                pass
                
            else:
                if detection_counter > 0:
                    detection_counter = 0
                    print("Цель потеряна, продолжаю поиск...")
                
                p = auv.get_pitch()
                r = auv.get_roll()
                
                if time.time() - self.mission_start_time > 4.5 and abs(p) < 0.02 and abs(r) < 0.02:
                    self.wall_maneuver()
                else:
                    y_err = self.target_yaw - auv.get_yaw()
                    self.drive(self.speed, y_err)
            
            cv2.imshow("Vision & Control", output_img)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        Mission().run()
    except KeyboardInterrupt:
        for i in range(5):
            auv.set_motor_power(i, 0)
        print("Программа остановлена")
