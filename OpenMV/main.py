import sensor
import image
import time
import pyb
import os, tf, math, uos, gc

# 初始化串口
uart = pyb.UART(3, 9600)

# 初始化摄像头
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

#宝藏判断相关参数
F_col = 1    #初始颜色识别，1红色，2蓝色
TorF = 0     #宝藏真假，0没宝藏，1真，2假
PRE_col = 0  #宝藏颜色，0没宝藏，1红，2蓝色

#目标检测相关参数
min_confidence = 0.3

#获取模型文件以及标签文件
net = tf.load("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
labels = [line.rstrip('\n') for line in open("labels.txt")]
'''
colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]
'''
# 开始时间
start_time = time.time()

#clock = time.clock() # 跟踪FPS帧率
while True:
    # 获取当前时间
    current_time = time.time()
    #clock.tick()
    img = sensor.snapshot()

    if current_time - start_time < 10:  # 在前两分钟内进行颜色识别
        FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
        uart.write(FH)
        continue
    else:  # 开始颜色和形状同步识别
        img = sensor.snapshot()
        for i, detection_list in enumerate(net.detect(img, thresholds=[(math.ceil(min_confidence * 255), 255)])):
            #print(clock.fps())
            if (i == 0):continue # background class
            if (len(detection_list) == 0):
                TorF = 0     #宝藏真假，0没宝藏，1真，2假
                PRE_col = 0  #宝藏颜色，0没宝藏，1红，2蓝色
                FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
                uart.write(FH)
                continue # no detections for this class?
            '''print("********** %s **********" % labels[i])
            for d in detection_list:
                [x, y, w, h] = d.rect()
                center_x = math.floor(x + (w / 2))
                center_y = math.floor(y + (h / 2))
                print('x %d\ty %d' % (center_x, center_y))
                img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)'''

            if (i == 1) :#----BC
                if (F_col==1):#判断初始颜色，1红色，2蓝色
                    TorF = 2     #宝藏真假，0没宝藏，1真，2假
                elif (F_col==2):
                    TorF = 1     #宝藏真假，0没宝藏，1真，2假
                PRE_col = 2  #宝藏颜色，0没宝藏，1红，2蓝色
                FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
                for a in range(0,8):
                    uart.write(FH)
                print('blue pre\n')
                break
            elif (i ==2 ):#----BT
                TorF = 2     #宝藏真假，0没宝藏，1真，2假
                PRE_col = 2  #宝藏颜色，0没宝藏，1红，2蓝色
                FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
                for a in range(0,8):
                    uart.write(FH)
                print('blue false pre\n')
                break
            elif (i==3) :#----RC
                TorF = 2     #宝藏真假，0没宝藏，1真，2假
                PRE_col = 1  #宝藏颜色，0没宝藏，1红，2蓝色
                FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
                for a in range(0,8):
                    uart.write(FH)
                print('red false pre\n')
                break
            elif(i == 4) :#----RT
                if (F_col == 1):#判断初始颜色，1红色，2蓝色
                    TorF = 1     #宝藏真假，0没宝藏，1真，2假
                elif (F_col == 2):
                    TorF = 2       #宝藏真假，0没宝藏，1真，2假
                PRE_col = 1  #宝藏颜色，0没宝藏，1红，2蓝色
                FH = bytearray([0x2C,0x12,F_col, TorF, PRE_col,0x5B,0x0d,0x0a])
                for a in range(0,8):
                    uart.write(FH)
                print('red pre\n')
                break


