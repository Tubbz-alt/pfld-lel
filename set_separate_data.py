import os
import numpy as np
import shutil
import cv2
from options.train_options import TrainOptions
from util.util import calculate_pitch_yaw_roll


class ImageSeparateData():
    def __init__(self, image_line, imageDir, image_size = 112):
        """
        0-195: landmark 坐标点  196-199: bbox 坐标点;
        200: 姿态(pose)         0->正常姿态(normal pose)          1->大的姿态(large pose)
        201: 表情(expression)   0->正常表情(normal expression)    1->夸张的表情(exaggerate expression)
        202: 照度(illumination) 0->正常照明(normal illumination)  1->极端照明(extreme illumination)
        203: 化妆(make-up)      0->无化妆(no make-up)             1->化妆(make-up)
        204: 遮挡(occlusion)    0->无遮挡(no occlusion)           1->遮挡(occlusion)
        205: 模糊(blur)         0->清晰(clear)                    1->模糊(blur)
        206: 图片名称
        """
        self.image_size = image_size
        self.image_repeat = 10
        """ 切片 空格"""
        image_line = image_line.strip().split()
        assert (len(image_line) == 207), '%s is not a valid image_line' % image_line
        self.image_line = image_line
        self.image_landmark = np.asarray(list(map(float, image_line[: 196])), dtype = np.float32).reshape(-1, 2)
        #print(self.image_landmark.shape)
        self.image_box = np.asarray(list(map(int, image_line[196 : 200])), dtype = np.int32)
        self.image_extreme_status = list(map(bool, image_line[200 : 206]))
        self.image_pose = self.image_extreme_status[0]
        self.image_expression = self.image_extreme_status[1]
        self.image_illumination = self.image_extreme_status[2]
        self.image_make_up = self.image_extreme_status[3]
        self.image_occlusion = self.image_extreme_status[4]
        self.image_blur = self.image_extreme_status[5]
        self.image_path = os.path.join(imageDir, image_line[206])
        self.image = None

        self.images = []
        self.image_landmarks = []
        self.image_boxs = []

    def save_data(self, path, prefix):
        attributes = [self.image_pose, self.image_expression, self.image_illumination, self.image_make_up, self.image_occlusion, self.image_blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        # 这些点的含义是每两个点代表脸上器官的边界，比如33，38 是右眉毛的边框点，看WFLW数据集脸谱图
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.images, self.image_landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix + '_' + str(i) + '.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels

    def load_data(self, is_train, mirror = None):
        if(mirror is not None):
            with open(mirror, 'r') as fp:
                image_lines = fp.readlines()
                assert len(image_lines) == 1
                mirror_idx = image_lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))

        # [98, 2] 数组算出宽高范围
        # 目的是直接找出人脸框？
        xy = np.min(self.image_landmark, axis=0).astype(np.int32)
        zz = np.max(self.image_landmark, axis=0).astype(np.int32)
        image_wh = zz - xy + 1

        image_center = (xy + image_wh / 2).astype(np.int32)
        boxsize = int(np.max(image_wh) * 1.2)
        xy = image_center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        #print(x1, y1, x2, y2)
        img = cv2.imread(self.image_path)
        image_height, image_width, _ = img.shape
        #print(img.shape)

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - image_width)
        edy = max(0, y2 - image_height)
        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        imgT = img[y1:y2, x1:x2]

        # copyMakeBorder 滤波插值扩充图像边缘像素
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.image_landmark + 0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()

        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.image_landmark - xy) / boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.images.append(imgT)
        self.image_landmarks.append(landmark)

        if is_train:
            while len(self.images) < self.image_repeat:
                angle = np.random.randint(-30, 30)
                center_x, center_y = image_center
                center_x = center_x + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                center_y = center_y + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = angle_rotate(angle, (center_x, center_y), self.image_landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))

                image_wh = np.ptp(landmark, axis = 0).astype(np.int32) + 1
                image_size = np.random.randint(int(np.min(image_wh)), np.ceil(np.max(image_wh) * 1.25))
                xy = np.asarray((center_x - image_size // 2, center_y - image_size // 2), dtype=np.int32)
                landmark = (landmark - xy) / image_size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + image_size
                image_height, image_width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - image_width)
                edy = max(0, y2 - image_height)
                x2 = min(image_width, x2)
                y2 = min(image_height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:, 0] = 1 - landmark[:, 0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.images.append(imgT)
                self.image_landmarks.append(landmark)



def angle_rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2, 3), dtype = np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1 - alpha) * center[0] - beta * center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta * center[0] + (1 - alpha) * center[1]

    landmark_ = np.asarray([(M[0, 0] * x + M[0 , 1] * y + M[0, 2],
                            M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmark])
    return M, landmark_


def get_dataset_list(imageDirs, saveDir, landmarkDir, is_train):
    with open(landmarkDir, 'r') as fp:
        image_lines = fp.readlines()
        labels = []
        save_img = os.path.join(saveDir, 'images')

        if not os.path.exists(save_img):
            os.mkdir(save_img)

        #image_lines = image_lines[: 100]
        for i, image_line in enumerate(image_lines):
            Img = ImageSeparateData(image_line, imageDirs)
            image_name = Img.image_path
            Img.load_data(is_train, Mirror_file)
            _, filename = os.path.split(image_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i) + '_' + filename)
            labels.append(label_txt)

    with open(os.path.join(saveDir, 'list.txt'), 'w') as fp:
        for label in labels:
            fp.writelines(label)


if __name__ == '__main__':
    opt = TrainOptions().parse()
    imageDirs = os.path.join(opt.dataroot, opt.image_dir)
    annotationsDirs = os.path.join(opt.dataroot, opt.annotations_dir)
    train_images = os.path.join(annotationsDirs, "list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt")
    test_images = os.path.join(annotationsDirs, "list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt")
    Mirror_file = os.path.join(annotationsDirs, "Mirror98.txt")
    landmarkDirs = [train_images, test_images]
    saveDirs = [opt.save_train_dir, opt.save_test_dir]
    #print(landmarkDirs)
    #print(saveDirs)
    for landmarkDir, saveDir in zip(landmarkDirs, saveDirs):
        saveDir = os.path.join(opt.dataroot, saveDir)
        if os.path.exists(saveDir):
            shutil.rmtree(saveDir)
        os.mkdir(saveDir)
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True

        imgs = get_dataset_list(imageDirs, saveDir, landmarkDir, is_train)
        print("%s if complete " % (landmarkDir))

    print("End separated")






