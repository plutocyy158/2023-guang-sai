import cv2
import time
import numpy as np
import torch
import heapq
import wiringpi


wiringpi.wiringPiSetup()
serial = wiringpi.serialOpen("/dev/ttyS4", 9600)#打开串口，通过查看手册获取编号
video = cv2.VideoCapture(11)#打开摄像头 摄像头编号不是1开始，建议一个一个试试....
video.set(cv2.CAP_PROP_FRAME_WIDTH,640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

blue_lower = np.array([100, 50, 50])
blue_upper = np.array([130, 255, 255])
red_lower = np.array([0, 50, 50])
red_upper = np.array([10, 255, 255])

blue_cx=0
blue_cy=0
red_cx=0
red_cy=0
x_units=0
y_units=0
next_0 = False
next_1 = False

# 处理图像
def process_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    global blue_cx,blue_cy,red_cx,red_cy,red_upper,red_lower,blue_upper,blue_lower,x_units,y_units

    # 蓝色色块检测
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 红色色块检测
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 寻找色块中心点并绘制
    for contour in blue_contours:
        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            blue_cx = int(M['m10'] / M['m00'])
            blue_cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (blue_cx, blue_cy), 5, (255, 0, 0), -1)

    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area > 100:
            M = cv2.moments(contour)
            red_cx = int(M['m10'] / M['m00'])
            red_cy = int(M['m01'] / M['m00'])
            cv2.circle(image, (red_cx, red_cy), 5, (0, 0, 255), -1)

    x_units = int((red_cx-blue_cx)/11)# 红色色块中心x坐标至蓝色色块中心x坐标像素点11等分平均值，作为单位量
    y_units = int((blue_cy-red_cy)/9)# 红色色块中心y坐标至蓝色色块中心y坐标像素点9等分平均值，作为单位量
            

    return image


def get_perspective_transform(image):
     # 定义源点（感兴趣区域的四个角点）
    src = np.float32([(blue_cx-x_units, red_cy-y_units), (blue_cx-x_units, blue_cy+y_units), (red_cx+x_units, red_cy-y_units), (red_cx+x_units, blue_cy+y_units)])
    
    # 定义目标点（映射感兴趣区域的坐标）
    dst = np.float32([[0, 0], [0, 360], [480, 0], [480, 360]])

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)

    # 应用透视变换到图像
    warped_image = cv2.warpPerspective(image, M, (480, 360))
    return warped_image

#路径规划
def get_neighbors(node, matrix):
    """
    获取节点的相邻节点
    """
    neighbors = []
    rows, cols = matrix.shape
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 右、左、下、上
    for dx, dy in directions:
        x, y = node
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and matrix[nx, ny] != 2:
            neighbors.append((nx, ny))
    return neighbors

def heuristic(node, goal):
    """
    估计从当前节点到目标节点的代价（欧几里得距离）
    """
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def astar(matrix, start, goal):
    """
    A*算法实现路径规划
    """
    rows, cols = matrix.shape
    visited = set()
    open_list = [(0, start)]  # 优先队列，元素为 (f, node)
    g_score = {start: 0}  # 起点到节点的实际代价
    f_score = {start: heuristic(start, goal)}  # 起点经过节点到目标节点的估计代价
    came_from = {}  # 记录每个节点的前驱节点

    while open_list:
        _, current = heapq.heappop(open_list)
        if current == goal:
            # 找到了路径，回溯重构路径
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        visited.add(current)

        neighbors = get_neighbors(current, matrix)
        for neighbor in neighbors:
            g = g_score[current] + 1  # 两个相邻节点的距离为1
            if neighbor in visited or g >= g_score.get(neighbor, np.inf):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = g
            f_score[neighbor] = g + heuristic(neighbor, goal)
            heapq.heappush(open_list, (f_score[neighbor], neighbor))

    # 无法到达终点，返回空路径
    return []

def planning (matrix,start,goal):
    path = []
    start_x,start_y = start
    goal_x,goal_y = goal
    if (20>start_x>0) and (20 > start_y > 0)and (20>goal_x>0) and (20>goal_y>0):
        matrix[goal_x][goal_y] = 1
        path = astar(matrix, start, goal)
        matrix[goal_x][goal_y] = 2
        return path
    else :
        return []

#(9,5)->(11,17)
def zheng_conversion(x1,y1):
    y2 = 2*x1-1
    x2 = 21-2*y1
    return x2,y2
#(11,17)->(9,5)
def ni_conversion(x2,y2):
    y1 = (21-x2)/2
    x1 = (y2+1)/2
    return x1,y1

#设置列表比较
def takeSecond(elem):
    return elem[1]

#地图矩阵
#1表示通路，2表示障碍，3表示起点，4表示终点
#matrix[y][x]-----注意和宝藏图坐标（x，y）一致，xy为宝藏图坐标
matrix = np.array([
    #0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,11,12,13,14,15,16,17,18,19,20----x
    [2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2, 2, 2, 2 ,2 ,2],#0
    [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1],#1
    [2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2],#2
    [2 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#3
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#4
    [2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#5
    [2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2],#6
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2],#7
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#8
    [2 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2],#9
    [2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2],#10
    [2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,2],#11
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#12
    [2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,2],#13
    [2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2],#14
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,2],#15
    [2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,2 ,2 ,1 ,2],#16
    [2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,2 ,1 ,2 ,1 ,2],#17
    [2 ,2 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2 ,1 ,2 ,2 ,2 ,1 ,2 ,1 ,2],#18
    [1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,2 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,2],#19
    [2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2 ,2, 2, 2, 2 ,2 ,2] #20-----y
])


if video.isOpened():#判断摄像头是否打开成功

    open, frame = video.read()
else:
    open = False

paishe_panduan = 0
start_time = time.time()
# 主程序
# 调试用
#while False:
# 测试流程用
while open:
    
    ret, image = video.read()
    if image is None:
        break
    if ret == True:

    # 处理图像
        processed_image = process_image(image)

        # 显示图像和坐标
        new_image = get_perspective_transform(processed_image)
        new_image = process_image(new_image)
        huaxian_image = np.copy(new_image)
        #绘制直线确保拍摄准确
        cv2.line(processed_image,(blue_cx-x_units, red_cy-y_units),(blue_cx-x_units, blue_cy+y_units),(0, 0, 0),3)
        cv2.line(processed_image,(blue_cx-x_units, red_cy-y_units),(red_cx+x_units, red_cy-y_units),(0, 0, 0),3)
        cv2.line(processed_image,(red_cx+x_units, blue_cy+y_units),(red_cx+x_units, red_cy-y_units),(0, 0, 0),3)
        cv2.line(processed_image,(red_cx+x_units, blue_cy+y_units),(blue_cx-x_units, blue_cy+y_units),(0, 0, 0),3)
        #画坐标点
        #cv2.line(huaxian_image,(blue_cx,blue_cy),(red_cx, blue_cy),(0, 0, 0),3)
        #cv2.line(huaxian_image,(blue_cx,blue_cy),(blue_cx, red_cy),(0, 0, 0),3)
        #cv2.line(huaxian_image,(blue_cx,red_cy),(red_cx, red_cy),(0, 0, 0),3)
        #cv2.line(huaxian_image,(red_cx,red_cy),(red_cx, blue_cy),(0, 0, 0),3)
        
        for b in range(0,11):#横
            cv2.line(huaxian_image,((blue_cx+int(x_units/2)),(red_cy-int(y_units/2))+(b*y_units)),((red_cx-int(x_units/2)), (red_cy-int(y_units/2))+(b*y_units)),(0, 0, 0),3)
        for v in range(0,11):#竖
            cv2.line(huaxian_image,((blue_cx+int(x_units/2))+(v*x_units),(blue_cy+int(y_units/2))),((blue_cx+int(x_units/2))+(v*x_units), (red_cy-int(y_units/2))),(0, 0, 0),3)
        cv2.putText(huaxian_image,"'Q'->paishe",(10,40),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,255),2)
        

        #results = model(new_image)
        #results_ = results.pandas().xyxy[0].to_numpy()

        cv2.namedWindow('Image',0)
        cv2.namedWindow('newimage',0)
        cv2.namedWindow('huaxian_image',0)
        cv2.imshow("Image", processed_image)
        cv2.imshow("newimage",new_image)
        cv2.imshow("huaxian_image",huaxian_image)
        cv2.resizeWindow('newimage',500,320)
        cv2.resizeWindow('Image',640,360)
        cv2.resizeWindow('huaxian_image',500,320)
        cv2.moveWindow('Image',0,0)
        cv2.moveWindow('newimage',650,0)
        cv2.moveWindow('huaxian_image',650,400)
        print(x_units,y_units)

        if cv2.waitKey(1) == ord('q'):
            cv2.imwrite('/home/orangepi/Desktop/test/image/saved_image.jpg', new_image)
            image_path = '/home/orangepi/Desktop/test/image/saved_image.jpg'
            test_image = cv2.imread(image_path)
            if test_image is not None:
                cv2.destroyAllWindows()
                cv2.imshow('Image_test', test_image)
                cv2.waitKey(5000)
                video.release()
                next_0 = True
                break
cv2.destroyAllWindows()


video.release()
model = torch.hub.load('/home/orangepi/yolov5','custom',path='/home/orangepi/Desktop/test/last.pt',source='local')
model.conf = 0.35

blue_cx=0
blue_cy=0
red_cx=0
red_cy=0

Treasure_coordinates = []#宝藏临时坐标
baozang_path = []#宝藏坐标
baozang_path_l_x = []#宝藏坐标 左下区域
baozang_path_r_x = []#宝藏坐标 右下区域
baozang_path_l_s = []#宝藏坐标 左上区域
baozang_path_r_s = []#宝藏坐标 右上区域

ditu_path = []#宝藏的地图坐标
ditu_path_l_x = [(1,1)]#宝藏的地图坐标 左下区域
ditu_path_r_x = [(1,1)]#宝藏的地图坐标 右下区域
ditu_path_l_s = [(1,1)]#宝藏的地图坐标 左上区域
ditu_path_r_s = [(1,1)]#宝藏的地图坐标 右上区域
plan_path = []

#调试用
while True:
#整体流程用
#while next_0:
    image_path = '/home/orangepi/Desktop/test/image/saved_image.jpg'
    image = cv2.imread(image_path)
    results = model(image)
    # 处理预测结果
    # 提取预测框、类别和置信度等信息
    results_ = results.pandas().xyxy[0].to_numpy()
    image = process_image(image)
    i = 0

    #预测框描绘
    for box in results_:
        l,t,r,b = box[:4].astype('int')
        confidence = str(round(box[4]*100,2))+"%"
        cls_name = box[6]

        #宝藏坐标转换---Treasure_coordinates[]
        center_x = (l+r)/2+5  # 获取圆心 x 坐标
        center_y = (t+b)/2-5 # 获取圆心 y 坐标
        converted_x = int((center_x-blue_cx+ 0)/x_units)
        converted_y = int((blue_cy-center_y- 0)/y_units)+1
        Treasure_coordinates.append((converted_x, converted_y))
        if converted_x < 6 and converted_y < 6:
            baozang_path_l_x.append((converted_x,converted_y))
        elif converted_x > 5 and converted_y < 6:
            baozang_path_r_x.append((converted_x,converted_y))
        elif converted_x < 6 and converted_y > 5:
            baozang_path_l_s.append((converted_x,converted_y))
        elif converted_x > 5 and converted_y > 5:
            baozang_path_r_s.append((converted_x,converted_y))
        else :
            pass

        #标注宝藏
        if cls_name == "baozang":
            i += 1
        else :
            pass
        cv2.rectangle(image,(l,t),(r,b),(0,255,0),2)
        cv2.putText(image,"(" + str(converted_x) + "," + str(converted_y) + ")",(l,t),cv2.FONT_ITALIC,1,(255,0,0),2)
    
    #删除多余数据并赋值排序至baozang_path，先从地图“左下区域”，到“右下区域”，然后至“右上区域”，最后是“左上区域”
    #baozang_path[0]----起点
    #baozang_path[1-2]--左下区域
    #baozang_path[3-4]--右下区域
    #baozang_path[5-6]--右上区域
    #baozang_path[7-8]--左上区域
    #baozang_path[9]----终点
    del Treasure_coordinates[8:],baozang_path_l_x[2:],baozang_path_r_x[2:],baozang_path_l_s[2:],baozang_path_r_s[2:],baozang_path[9:]
    baozang_path.append((3,4))
    baozang_path = baozang_path + baozang_path_l_x + baozang_path_r_x + baozang_path_r_s + baozang_path_l_s
    baozang_path.append((9,8))
    #print("baozang_path",baozang_path)

    #宝藏图坐标转换地图坐标
    for a in baozang_path:
        x1,y1 = a
        x2,y2 = zheng_conversion(x1,y1)
        ditu_path.append((x2,y2))
        #baozang_path.clear()

    
    cv2.imwrite('/home/orangepi/Desktop/test/image/new_image.jpg', image)
    print(ditu_path)
    print(len(ditu_path))
    #设定宝藏为障碍点
    if(len(ditu_path)<3):
        break
    else:
        ditu_path_l_x = [ditu_path[1]]+[ditu_path[2]]#宝藏的地图坐标 左下区域
    if(len(ditu_path)<5):
        break
    else:
        ditu_path_r_x = [ditu_path[3]]+[ditu_path[4]]#宝藏的地图坐标 右下区域
    ditu_path_l_x.sort(key=takeSecond)
    ditu_path_r_x.sort(key=takeSecond)
    if(len(ditu_path)<7):
        ditu_path_r_s.append((5,17))
        break
    else:
        ditu_path_r_s = [ditu_path[5]]+[ditu_path[6]]#宝藏的地图坐标 左上区域
    if(len(ditu_path)<9):
        ditu_path_l_s.append((5,17))
        break
    else:
        ditu_path_l_s = [ditu_path[7]]+[ditu_path[8]]#宝藏的地图坐标 右上区域
    
    ditu_path_l_x.sort(key=takeSecond)
    ditu_path_r_x.sort(key=takeSecond)
    '''
    print("ditu_path_l_x",ditu_path_l_x)
    print("ditu_path_r_x",ditu_path_r_x)
    print("ditu_path_r_s",ditu_path_r_s)
    print("ditu_path_l_s",ditu_path_l_s)
    '''
    baozang_path.clear()
    
    
    for plan in ditu_path:
        c,d =plan
        if (c == 13 and d == 5) or (c == 7 and d == 17):
            continue
        matrix[c][d] = 2
    
    # 显示图像
    cv2.imwrite('/home/orangepi/Desktop/test/image/new_image.jpg', image)
    cv2.putText(image,"Part2",(5,30),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,255),1)
    cv2.imshow('Image', image)
    cv2.waitKey(5000)
    next_1 = True
    break
    '''
    if cv2.waitKey(1) == 27:
        break
    '''
cv2.destroyAllWindows()

UART_data = []#串口接收数据，0,1,7数据为握手数据
car_place = (20,20)#小车坐标点数据，发送为（x，y），运用为（y，x）
end_path = [(0,0)]
colours = 2 #初始颜色识别，1红色，2蓝色
baozang_colours = 3 #宝藏颜色，0没宝藏，1红，2蓝色
baozang_zhenjia = 4 #宝藏真假，0没宝藏，1真，2假
zhen_baozang = 0
wei_baozang = 0
jia_baozang = 0
baozang_sign_lx = 0 #左下区域宝藏已寻找标志位
baozang_sign_rx = 0 #右下区域宝藏已寻找标志位
baozang_sign_ls = 0 #左上区域宝藏已寻找标志位
baozang_sign_rs = 0 #右上区域宝藏已寻找标志位
duqu_panduan = 0
UART_count = 1
sign_0 = 0
reset = 0
x_now = 0
y_now = 0
x_once = 0
y_once = 0
x_fut = 0
y_fut = 0
end_path = [(0,0)]
fuzhi_sign = 1
bef_UART_data = [1,1,1,1,1,1,1,1]
new_image_path = '/home/orangepi/Desktop/test/image/new_image.jpg'


wiringpi.serialFlush(serial)
#处理完毕，开始与智能车通信
#调试用
while True:
#整体流程用
#while next_1:
    image1 = cv2.imread(new_image_path)
    cv2.putText(image1,"Part3",(5,30),cv2.FONT_HERSHEY_COMPLEX,1.2,(0,255,255),1)
    cv2.imshow('Image', image1)
    
    while wiringpi.serialDataAvail(serial):
        data = wiringpi.serialGetchar(serial)
        UART_data.append(data)
        if UART_data[0] != 82 :
            UART_data.clear()
        if len(UART_data) == 8 :
            if len(bef_UART_data) == 8 :
                if ((UART_data[0] == 82) and (UART_data[1] == 33) and (UART_data[7] == 242)) and (bef_UART_data != UART_data):
                    print("接收的数据为：",UART_data)
                    x_now = UART_data[2];y_now = UART_data[3];#x_once = UART_data[7];y_once = UART_data[8]
                    car_place = (21-2*UART_data[3],2*UART_data[2]-1)
                    colours = UART_data[4]
                    if ((UART_data[5] == 0 and UART_data[6] != 0) or (UART_data[5] != 0 and UART_data[6] == 0)) and (UART_data[2] != 3 and UART_data[3] != 4):#接受异常202,220,赋值为假宝藏。待删除
                        '''
                        if colours == 1:
                            baozang_colours = 2
                        else:
                            baozang_colours = 1
                        baozang_you = True
                        '''
                        print("异常接收")
                        UART_data.clear()
                        continue
                    elif UART_data[5] != 0:
                        baozang_colours = UART_data[5]
                        if  UART_data[6] != 0:
                            baozang_you = True
                        else:
                            baozang_you = False 
                    else:
                        baozang_you = False
                    baozang_zhenjia = UART_data[6]
                    bef_UART_data = UART_data.copy()
                    UART_data.clear()
                    fuzhi_sign = 0
                    UART_count += 1
                    # 已经找到的宝藏累加
                    if (colours == baozang_colours) and baozang_zhenjia == 1:
                        zhen_baozang += 1
                    elif (colours == baozang_colours) and baozang_zhenjia == 2:
                        wei_baozang += 1
                    elif colours != baozang_colours:
                        jia_baozang += 1
                    wiringpi.serialFlush(serial)
                    break
                else :
                    wiringpi.serialFlush(serial)
                    UART_data.clear()
                        
            else :
                wiringpi.serialFlush(serial)
                UART_data.clear()

    
    
    
    #----第一区域寻找
    if ((colours == baozang_colours) and baozang_zhenjia == 0) and (jia_baozang == 0 and wei_baozang == 0 and zhen_baozang == 0)  :
        #找左下区域第一个宝藏
        end_path[0] = ditu_path_l_x[0]
        baozang_sign_lx = 1
    
    elif baozang_sign_lx ==1 and baozang_sign_rx == 0 and baozang_you :
        if (colours != baozang_colours) :
            #左下区域第一个宝藏为假,但不是“伪宝藏”，寻找左下区域第二个宝藏 ditu_path_l_x
            baozang_sign_lx = 1
            #将左下区域第一个宝藏的中心对称坐标，定义为右上区域待寻找宝藏，将其留下 ditu_path_r_s
            x_tran,y_tran = end_path[0]
            x_tran = int(2*10-x_tran)
            y_tran = int(2*10-y_tran)
            if (x_tran,y_tran) in ditu_path_r_s:
                index = ditu_path_r_s.index((x_tran,y_tran))
                linshi_path = ditu_path_r_s[index]
                ditu_path_r_s.clear()
                ditu_path_r_s = [linshi_path]

            else:
                pass
            #发送路径规划，理论上，上下无关联
            end_path[0] = ditu_path_l_x[1]
            baozang_you = False

        elif (colours == baozang_colours) :
            #左下区域第一个宝藏为伪宝藏/真宝藏ditu_path_l_x[0] 去右下区域第一个宝藏 ditu_path_r_x[0]
            baozang_sign_lx = 2
            #将左下区域第一个宝藏的中心对称坐标移除，去找另一坐标 ditu_path_r_s
            x_tran,y_tran = end_path[0]
            x_tran = 2*10-x_tran
            y_tran = 2*10-y_tran
            if (x_tran,y_tran) in ditu_path_r_s:
                ditu_path_r_s.remove((x_tran,y_tran))
                
            else :
                pass
            #发送路径规划，理论上，上下无关联
            end_path[0] = ditu_path_r_x[0]
            baozang_sign_rx = 1
            baozang_you = False

    #----第二区域寻找
    elif (baozang_sign_lx ==2 and baozang_sign_rx==1) and baozang_you:
        if (colours != baozang_colours) :
            #右下区域第一个宝藏为假，寻找右下区域第二个宝藏 ditu_path_l_x
            baozang_sign_rx = 1
            #将右下区域第一个宝藏的中心对称坐标，定义为左上区域待寻找宝藏，将其留下 ditu_path_r_s
            x_tran,y_tran = end_path[0]
            x_tran = 2*10-x_tran
            y_tran = 2*10-y_tran
            if (x_tran,y_tran) in ditu_path_l_s:
                index = ditu_path_l_s.index((x_tran,y_tran))
                linshi_path = ditu_path_l_s[index]
                ditu_path_l_s.clear()
                ditu_path_l_s = [linshi_path]
            else:
                pass
            baozang_you = False
            #发送路径规划，理论上，上下无关联su
            end_path[0] = ditu_path_r_x[1]

        elif (colours == baozang_colours) :
            #右下区域第一个宝藏为伪宝藏/真宝藏， ditu_path_l_x
            baozang_sign_rx = 2
            #将右下区域第一个宝藏的中心对称坐标移除，去找另一坐标 ditu_path_r_s
            x_tran,y_tran = end_path[0]
            x_tran = 2*10-x_tran
            y_tran = 2*10-y_tran
            if (x_tran,y_tran) in ditu_path_l_s:
                ditu_path_l_s.remove((x_tran,y_tran))
            else:
                pass
            baozang_you = False
            #发送路径规划，理论上，上下无关联
            end_path[0] = ditu_path_r_s[0]

    #----第三区域、第四区域寻找
    elif (baozang_sign_lx ==2 and baozang_sign_rx==2)and baozang_you:
        if (zhen_baozang + wei_baozang == 2):
            x_tran,y_tran = end_path[0]
            x_tran = 2*10-x_tran
            y_tran = 2*10-y_tran
            if (x_tran,y_tran) in ditu_path_l_s:
                ditu_path_l_s.remove((x_tran,y_tran))
            else:
                pass
            baozang_you = False
            end_path[0] = ditu_path_r_s[0]

        elif zhen_baozang == 3 :
            end_path[0] = ditu_path[9]

        elif (wei_baozang + zhen_baozang == 3) :
            end_path[0] = ditu_path_l_s[0]

        elif (wei_baozang + zhen_baozang > 3) :
            end_path[0] = ditu_path[9]

        else :
            pass
        baozang_you = False

    else :
        pass
    '''
    print("ditu_path_l_x",ditu_path_l_x)
    print("ditu_path_r_x",ditu_path_r_x)
    print("ditu_path_r_s",ditu_path_r_s)
    print("ditu_path_l_s",ditu_path_l_s)
    '''
    
    if baozang_sign_lx and UART_count:
        time.sleep(0.01)
        print("end_path:",end_path)
        UART_count = 0
        if car_place != end_path[0] :
            plan_path = planning(matrix,car_place,end_path[0])
            if plan_path :
                x_fut,y_fut = plan_path[2]
                x_fut,y_fut = ni_conversion(x_fut,y_fut)
                x_fut = int(x_fut)
                y_fut = int(y_fut)
                #print("ditu_path",ditu_path)
                for k in range(0,30):
                    wiringpi.serialPutchar(serial, 0x52)
                    wiringpi.serialPutchar(serial, 0x21)
                    wiringpi.serialPutchar(serial, x_fut)
                    wiringpi.serialPutchar(serial, y_fut)
                    wiringpi.serialPutchar(serial, 0xf2)
                    wiringpi.serialPutchar(serial, 0x0d)
                    wiringpi.serialPutchar(serial, 0x0a)
                print("end_path:",end_path)
                print("x:",x_fut)
                print("y:",y_fut)
                print("真宝藏个数：",zhen_baozang)
                print("伪宝藏个数：",wei_baozang)
                print("baozang_sign_lx",baozang_sign_lx)
                print("baozang_sign_rx",baozang_sign_rx)
                #print("ditu_path_l_s",ditu_path_l_s)
                #print("ditu_path_r_s",ditu_path_r_s)
        else:#起点等于终点，发送（0,0）标志位，小车返回伪宝藏
            print("起点等于终点,已发送(ff,ff)")
            for k in range(0,30):
                wiringpi.serialPutchar(serial, 0x52)
                wiringpi.serialPutchar(serial, 0x21)
                wiringpi.serialPutchar(serial, 0xff)
                wiringpi.serialPutchar(serial, 0xff)
                wiringpi.serialPutchar(serial, 0xf2)
                wiringpi.serialPutchar(serial, 0x0d)
                wiringpi.serialPutchar(serial, 0x0a)
        

    if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
