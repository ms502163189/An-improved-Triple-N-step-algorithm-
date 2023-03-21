import os
import time
from pathlib import Path 
import numpy as np
import cv2
import pygame
import random
import pathlib

from config import config
from projector import Projector
from camera import HK_Camera
from fringes import Fringes

class ImageReader():
    def __init__(self, config):
        self._c = config

    def read(self, case, gamma1):
        root = pathlib.Path(self._c.measure_path)
        images = []
        list = [7, 4, 3]
        imgs_f1 = []
        imgs_f2 = []
        imgs_f3 = []
        for ind in range(14):
            name_h = root / f"{case:0>2d}{ind:0>2d}h.bmp"
            print(name_h)
            img = cv2.imread(name_h.as_posix(), 0)
            height, width = img.shape[:2]
            # print("height, width",height)
            # print("width",width)

            # start_row, start_col = 0, 0
            # end_row, end_col = height, width
            start_row, start_col = 490, 0
            end_row, end_col = 540, width

            cropped = img[start_row:end_row, start_col:end_col]  # 将图像裁剪成1920 * 50
            # slight gamma correction to avoid outliers
            cropped = np.power(cropped / 255., gamma1) * 255
            cropped = cv2.blur(cropped, (7, 7))
            if ind < 7:
                print("f1")
                imgs_f1.append(cropped)
            elif ind < 11:
                print("f2")
                imgs_f2.append(cropped)
            elif ind < 14:
                print("f3")
                imgs_f3.append(cropped)
        images.append(imgs_f1)
        images.append(imgs_f2)
        images.append(imgs_f3)
        print("len(images[0])", len(images[0]))
        print("len(images[1])", len(images[1]))
        print("len(images[2])", len(images[2]))

        name = f"train_data{case:03d}"
        path = os.path.join(self._c.net_dir, name)
        np.savez(path, image1=images[0], image2=images[1], image3=images[2])
        return case

class Capture():
    def __init__(self, config):
        self._c = config
        self.prj = Projector(config)
        self.hkc = HK_Camera()
        self.hkc.start()

    def capture_one(self, path, hv="h"):
            i = 0
            prj_flag = True
            while prj_flag:
                prj_flag = self.prj.update(hv)
                # print("prj_flag", prj_flag)
                time.sleep(32/96)
                self.prj.black()
                self.hkc.capture_one(name=path + f"{i:0>2d}{hv}.bmp")
                i+=1
    #标定时候用这个函数，投影图需要制作
    def calibra_capture(self):
        count = 15
        root = Path(self._c.calibra_path)
        root.mkdir(parents=True, exist_ok=True)
        while self.prj.wait_to_begin():
            self.capture_one(str(root)+ f"/{count:0>2d}_", hv="h")
            self.capture_one(str(root)+ f"/{count:0>2d}_", hv="v")
            count +=1

    #测量实验时候用这个函数，12步相移的投影图需要通过fringes.py生成
    def measure_capture(self):
        count = 0
        root = Path(self._c.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        while self.prj.wait_to_begin():
            self.capture_one(str(root)+ f"/{count:0>2d}", hv="h")
            count +=1

    def measure_capture_train_data(self, num, data_num):  #生成训练时候的数据集
        count = 0
        flag = 0
        root = Path(self._c.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        while self.prj.wait_to_begin():   #按空格就拍
            for i in range(num):           #按一次空格拍100组
                # print("pattern_path",self._c.pattern_path)
                # print("steps",self._c.steps)
                # print("cfg.measure_path ",self._c.measure_path )
                self._c.gamma = random.uniform(0.8,2.0)
                print("self._c.gamma ", self._c.gamma)
                self._c.hv="h"
                Fringes(self._c).save_images()
                self._c.hv="v"
                Fringes(self._c).save_images()
                self.capture_one(str(root)+ f"/{count:0>2d}", hv="h")
                case = ImageReader(self._c).read(case=count, gamma1=1.25)  # gamma1是对图像进行略微的gamma矫正
                print("{} 组已经保存".format(case))
                count +=1
                if count % 10 == 0:
                    del_files(root)
                if count == data_num:
                    flag = 1
                    break
            if flag == 1:
                break
        return count


    def exit(self):
        self.prj.exit()
        self.hkc.exit()

    def exit1(self):
        self.hkc.exit()
def wait_to_begin():
    pygame.init()
    wait_proj = True
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    # self.screen = pygame.display.set_mode((0,0))

    screen.fill((0, 0, 255))
    bg = pygame.image.load('./data/cover.png').convert()
    blank = pygame.image.load('./data/blank.png').convert()
    clock = pygame.time.Clock()  # 设置时钟
    clock.tick(24)

    screen.blit(bg,(0,0))
    pygame.display.flip()
    pygame.display.update()
    while True:
        clock.tick(24)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                wait_proj = False
                exit()

        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_SPACE]:
            print("Begin to project patterns...")
            break

        if keys_pressed[pygame.K_ESCAPE]:
            wait_proj = False
            print("Projection finished.\n")
            break
    return wait_proj

def dir_is_or_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    from config import config

    cfg = config()
    # cap.calibra_capture() # 标定的时候用这个
    # input = input("生成训练数据集还是测试数据集：")
    # if (input == "test"):
    count = 0
    while wait_to_begin():

        cfg.steps = [9, 9, 9]
        cfg.pattern_path = "./data/patterns/999-step/"
        cfg.measure_path = "./data/recordings/2/9-step/"
        dir_is_or_not_exist(cfg.measure_path)
        cap = Capture(cfg)
        root = Path(cfg.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        print("root", root)
        cap.capture_one(str(root) + f"/{count:0>2d}", hv="h")  # 20-step
        cap.exit1()
        #
        cfg.steps = [20, 20, 20]
        cfg.pattern_path = "./data/patterns/202020-step/"
        cfg.measure_path = "./data/recordings/2/20-step/"
        dir_is_or_not_exist(cfg.measure_path)
        # print("cfg.measure_path  ",cfg.measure_path )
        cap = Capture(cfg)
        root = Path(cfg.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        print("root", root)
        cap.capture_one(str(root) + f"/{count:0>2d}", hv="h")  # 20-step
        cap.exit1()

        cfg.steps = [12, 12, 12]
        cfg.pattern_path = "./data/patterns/121212-step/"
        cfg.measure_path = "./data/recordings/2/12-step/"
        dir_is_or_not_exist(cfg.measure_path)
        cap = Capture(cfg)
        root = Path(cfg.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        print("root", root)
        cap.capture_one(str(root) + f"/{count:0>2d}", hv="h")  # 20-step
        cap.exit1()

        cfg.steps = [3, 3, 3]
        cfg.pattern_path = "./data/patterns/333-step/"
        cfg.measure_path = "./data/recordings/2/3-step/"
        dir_is_or_not_exist(cfg.measure_path)
        cap = Capture(cfg)
        root = Path(cfg.measure_path)
        root.mkdir(parents=True, exist_ok=True)
        print("root", root)
        cap.capture_one(str(root) + f"/{count:0>2d}", hv="h")  # 20-step
        cap.exit1()
        count += 1
    cap.exit()





