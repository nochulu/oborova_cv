import cv2
import mss
import pyautogui
import numpy as np

# Отключение встроенной задержки pyautogui
pyautogui.PAUSE = 0

# 1. КООРДИНАТЫ ИГРОВОЙ ОБЛАСТИ
GAME_REGION = {"top": 250, "left": 590 , "width": 600, "height": 180}
  
# 2. НАСТРОЙКИ КОЛЛАЙДЕРОВ (y1, y2, x1, x2)
BOX_TOP = {"y1": 100, "y2": 120, "x1": 90, "x2": 120}     # Верхний коллайдер
BOX_FRONT = {"y1": 142, "y2": 152, "x1": 105 , "x2": 125}   # Для короткого прыжка
BOX_AUX = {"y1": 142, "y2": 152, "x1": 125,  "x2": 175}     # Для длинного прыжка
  
# Настройки скорости
SPEED_MULTIPLIER = 0.5  # Скорость передних коллайдеров
MAX_SHIFT = 250         # Максимальное смещение передних коллайдеров

def draw_colliders(img, shift, top_hit, front_hit, aux_hit):
    """Отрисовка отладочной информации через OpenCV"""
    color_top = (0, 0, 255) if top_hit else (0, 255, 0)
    color_front = (0, 0, 255) if front_hit else (255, 0, 0)
    color_aux = (0, 0, 255) if aux_hit else (0, 255, 255)

    cv2.rectangle(img, (BOX_TOP['x1'], BOX_TOP['y1']), 
                  (BOX_TOP['x2'], BOX_TOP['y2']), color_top, 2)
    cv2.rectangle(img, (BOX_FRONT['x1'] + shift, BOX_FRONT['y1']), 
                  (BOX_FRONT['x2'] + shift, BOX_FRONT['y2']), color_front, 2)
    cv2.rectangle(img, (BOX_AUX['x1'] + shift, BOX_AUX['y1']), 
                  (BOX_AUX['x2'] + shift, BOX_AUX['y2']), color_aux, 2)
    
    cv2.imshow("Dino Vision (Press 'q' in this window to quit)", img)

def play_game():
    print("Скрипт запустится через 3 секунды. ")
    pyautogui.sleep(3)
    pyautogui.press('space') # Запуск игры
    
    # Используем тики процессора через OpenCV
    start_tick = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()
    is_ducking = False

    with mss.mss() as sct:
        while True:
            # Расчет смещения коллайдеров из-за роста скорости игры
            elapsed_time = (cv2.getTickCount() - start_tick) / tick_freq
            shift = int(min(elapsed_time * SPEED_MULTIPLIER, MAX_SHIFT))

            # Захват экрана и конвертация для OpenCV
            screen = np.array(sct.grab(GAME_REGION))
            
            gray = cv2.cvtColor(screen, cv2.COLOR_BGRA2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            top_roi = edges[BOX_TOP['y1']:BOX_TOP['y2'], 
                            BOX_TOP['x1'] : BOX_TOP['x2']]
            front_roi = edges[BOX_FRONT['y1']:BOX_FRONT['y2'], 
                              BOX_FRONT['x1'] + shift : BOX_FRONT['x2'] + shift]
            aux_roi = edges[BOX_AUX['y1']:BOX_AUX['y2'], 
                            BOX_AUX['x1'] + shift : BOX_AUX['x2'] + shift]

            top_hit = cv2.countNonZero(top_roi) > 0
            front_hit = cv2.countNonZero(front_roi) > 0
            aux_hit = cv2.countNonZero(aux_roi) > 0

            # ЛОГИКА УПРАВЛЕНИЯ
            if front_hit:
                if is_ducking:
                    pyautogui.keyUp('down')
                    is_ducking = False
                
                if aux_hit:
                    # Длинный прыжок
                    pyautogui.keyDown('space')
                    pyautogui.sleep(0.18)  
                    pyautogui.keyUp('space')
                    pyautogui.sleep(0.01) 
                     
                    pyautogui.keyDown('down')
                    pyautogui.sleep(0.01)  
                    pyautogui.keyUp('down')
                else:
                    # Короткий прыжок
                    pyautogui.keyDown('space')
                    pyautogui.sleep(0.1) 
                    pyautogui.keyUp('space')
                    pyautogui.sleep(0.01) 
                    
                    pyautogui.keyDown('down')
                    pyautogui.sleep(0.01)  
                    pyautogui.keyUp('down')

            elif top_hit and not front_hit:
                if not is_ducking:
                    pyautogui.keyDown('down')
                    is_ducking = True
            else:
                if is_ducking:
                    pyautogui.sleep(0.01)
                    pyautogui.keyUp('down')
                    is_ducking = False

            debug_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            draw_colliders(debug_img, shift, top_hit, front_hit, aux_hit)

            # Выход по нажатию 'q' с активным окном отладки
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Скрипт остановлен.")
                break

    if is_ducking:
        pyautogui.keyUp('down')
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()