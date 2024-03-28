import cv2
import time
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

pipeline = rs.pipeline()
config = rs.config()
# config.enable_device('Your_D455_Device_Serial_Number')
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
#config.enable_stream(rs.stream.gray, 640, 480, rs.format.bgr8, 60)
# 启动流


profile = pipeline.start(config)
# 获取相机设备
print(1)
device = profile.get_device()
print(1)
# 检查是否有数据流
if device.is_streaming():
    print("相机已启动，有数据流传入。")
else:
    print("相机启动失败，没有数据流传入。")


config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
pipeline.start(config)

align_to = rs.stream.color  #与color流对齐
align = rs.align(align_to)

# 创建视频写入器
fps = 30.0  # 帧率
width = 640  # 视频宽度
height = 480  # 视频高度
output_filename = 'output108.mp4'  # 输出视频文件名
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
pTime = 0
count = 0
first_condition = 0
second_condition = 0
third_condition = 0
flat_x = []
flat_y = []
lmlist = []
d1 = 0
d2 = 0
x = 0
y = 0
cls = -1
model = YOLO("./50epoch2/best.pt")
try:
    while True:
        # 等待获取帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)  # 获取对齐帧
        depth_frames = aligned_frames.get_depth_frame()
        if not depth_frames:
            continue
        depth_image = np.asarray(depth_frames.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 提取RGB图像帧
        #color_frame = frames.get_color_frame()
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            print('not color')
            continue

        # 将RGB图像帧转换为numpy数组


        color_image = np.asanyarray(color_frame.get_data())
        img = color_image.copy()
        #img1 = color_image.copy()
        detect_results = model(img)
        if detect_results[0].boxes.shape[0] == 0:
            pass
        else:
            x1 = int(detect_results[0].boxes[0].xyxy[0][0].cpu().numpy().astype(np.float64))
            y1 = int(detect_results[0].boxes[0].xyxy[0][1].cpu().numpy().astype(np.float64))
            x2 = int(detect_results[0].boxes[0].xyxy[0][2].cpu().numpy().astype(np.float64))
            y2 = int(detect_results[0].boxes[0].xyxy[0][3].cpu().numpy().astype(np.float64))
            cls = detect_results[0].boxes[0].cls[0].cpu().numpy().astype(np.int64)
            conf = detect_results[0].boxes[0].conf[0].cpu().numpy().astype(np.float64)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            #cv2.putText(img, str(conf), (x1, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(img, str(cls), (x1 - 20, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            x=int(x1/2+x2/2)
            y=int(y1/2+y2/2)
        results = pose.process(img)
        
        if results.pose_landmarks:
            #mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for index, lm in enumerate(results.pose_landmarks.landmark):
                # 保存每帧图像的宽、高、通道数
                h, w, c = img.shape

                # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
                # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
                if lm.x<0 or lm.x>1 or lm.y<0 or lm.y>1:
                    lm.x = 0.5
                    lm.y = 0.5
                cx, cy = int(lm.x * w), int(lm.y * h)

                if index ==12:
                    d1 = depth_frames.get_distance(cx, cy)
                    # 打印坐标信息
                if index ==11:
                    d2 = depth_frames.get_distance(cx, cy)

                #print(d1/2+d2/2)
                cv2.putText(img, str('%.4f' %(d1/2+d2/2)), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                y5=results.pose_landmarks.landmark[15].y-results.pose_landmarks.landmark[13].y
                y6=results.pose_landmarks.landmark[16].y-results.pose_landmarks.landmark[14].y
                x5=results.pose_landmarks.landmark[15].x-results.pose_landmarks.landmark[13].x
                x6=results.pose_landmarks.landmark[16].x-results.pose_landmarks.landmark[14].x 
                
                y7=results.pose_landmarks.landmark[23].y-results.pose_landmarks.landmark[25].y
                y8=results.pose_landmarks.landmark[24].y-results.pose_landmarks.landmark[26].y
                x7=results.pose_landmarks.landmark[23].x-results.pose_landmarks.landmark[25].x
                x8=results.pose_landmarks.landmark[24].x-results.pose_landmarks.landmark[26].x 
                if cls ==0:
                    #img = cv2ImgAddText(img, "军人", x1 - 20, y1, (0, 0, 139), 20)
                    if  x5==0:
                        x5 =0.001
                    if x6 ==0:
                        x6 =0.001
                    if int(abs(y5)/abs(x5))>1.3 and int(abs(y6)/abs(x6))>1.3  :
                        #img = cv2ImgAddText(img, "投降", x1+20, y1, (0, 0, 139), 20)
                        cv2.putText(img, 'white', (x1+20, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 25), 1)
                    else:
                        #img = cv2ImgAddText(img, "攻击", x1 + 20, y1, (0, 0, 139), 20)
                        cv2.putText(img, 'attack', (x1+20, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                if cls ==1:
                    #img = cv2ImgAddText(img, "平民", x1 - 20, y1, (0, 255, 0), 20)
                    if  y7==0:
                        y7 =0.001
                    if y8 ==0:
                        y8 =0.001
                    if int(abs(x7)/abs(y7))>1 or int(abs(x8)/abs(y8))>1 :
                        #img = cv2ImgAddText(img, "受伤", x1 + 20, y1, (0, 255, 0), 20)
                        cv2.putText(img, 'injured', (x1+20, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                    else:
                        #img = cv2ImgAddText(img, "正常", x1 + 20, y1, (0, 255, 0), 20)
                        cv2.putText(img, 'normal', (x1+20, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (25, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        #color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        # 显示RGB图像
        cv2.imshow('RGB Image', img)
        #cv2.imshow("detect image:", img1)

        # 写入视频帧
        out.write(img)

        # 检测按键，按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 关闭视频写入器和相机流
    out.release()
    cv2.destroyAllWindows()
    pipeline.stop()



# ------------------------------------------------
# mpDraw = mp.solutions.drawing_utils
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
#
# cap = cv2.VideoCapture(0)
# success, img = cap.read()
# video_size = (img.shape[1], img.shape[0])
#
# pTime = 0
# count = 0
# first_condition = 0
# second_condition = 0
# third_condition = 0
# flat_x = []
# flat_y = []
# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = pose.process(imgRGB)
#     #print(results.pose_landmarks)
#     # if results.pose_landmarks:
#     #     mpDraw.draw_landmarks(img, results.pose_landmarks,
#     #                           mpPose.POSE_CONNECTIONS)
#
#     cTime = time.time()
#     fps = 1/(cTime-pTime)
#     pTime = cTime
#     cv2.putText(img, str(int(fps)), (25, 50), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 0), 3)
#
#     cv2.imshow("Image", img)
#     cv2.waitKey(25)
#     if cv2.waitKey(1) == 27:
#         break
#
# cap.release()
#
# cv2.destroyAllWindows()
