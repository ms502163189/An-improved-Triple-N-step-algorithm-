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

def measure_capture(cfg,count):
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

def dir_is_or_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    from config import config

    cfg = config()
    # # 标定的时候用下面三行
    # cap = Capture(cfg)
    # cap.calibra_capture()
    # cap.exit()

    # 测试的时候用下面两行
    count = 0
    measure_capture(cfg,count)





