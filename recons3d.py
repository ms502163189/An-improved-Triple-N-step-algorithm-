import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from phase import *
from calibration import Calibrator
from config import config

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

def phase_average(phi1, phi2, phi3):
    phi_r1 = (phi1+phi2+phi3)/3
    phi_r2 = phi_r1+2*np.pi/3
    phi_r3 = phi_r1-2*np.pi/3

    err1 = np.minimum(np.mod(phi_r1 - phi2, 2*np.pi), np.mod(phi2-phi_r1, 2*np.pi))
    err2 = np.minimum(np.mod(phi_r2 - phi2, 2*np.pi), np.mod(phi2-phi_r2, 2*np.pi))
    err3 = np.minimum(np.mod(phi_r3 - phi2, 2*np.pi), np.mod(phi2-phi_r3, 2*np.pi))

    phi = phi_r1
    mask2 = (err2<err1) * (err2<err3)
    phi[mask2] = phi_r2[mask2]
    mask3 = (err3<err1) * (err3<err2)
    phi[mask3] = phi_r3[mask3]
    return phi


def unwrapped_phase(cfg, psi):
    phi1, T1 = PE(cfg).phase_unwarping2(psi[0],psi[1],cfg.Tc[0],cfg.Tc[1])
    phi1 = phi1*T1/cfg.Tc[0]
    phi1 = phi1.astype(np.float32)
    return phi1,T1

def old_Triple_N_step(i,cfg,fringe_image):
    img1_group0 = []
    img1_group1 = []
    img1_group2 = []
    if i == 1:
        name = 0
    if i == 2:
        name = 1
    print(name)
    img1_0, img1_3, img1_6 = fringe_image[name][11], fringe_image[name][3], fringe_image[name][7]
    img1_group0.append(img1_0)
    img1_group0.append(img1_3)
    img1_group0.append(img1_6)
    img1_1, img1_4, img1_7 = fringe_image[name][0], fringe_image[name][4], fringe_image[name][8]
    img1_group1.append(img1_1)
    img1_group1.append(img1_4)
    img1_group1.append(img1_7)
    img1_2, img1_5, img1_8 = fringe_image[name][1], fringe_image[name][5], fringe_image[name][9]
    img1_group2.append(img1_2)
    img1_group2.append(img1_5)
    img1_group2.append(img1_8)

    phase_measures0 = PE(cfg).psi_extract(images=img1_group0)  # 用origin求包裹相位
    phase_measures1 = PE(cfg).psi_extract(images=img1_group1)  # 用2pai/3N求包裹相位
    phase_measures2 = PE(cfg).psi_extract(images=img1_group2)  # 用4pai/3N求包裹相位

    return phase_measures0,phase_measures1,phase_measures2,img1_group0,img1_group1,img1_group2

def new_Triple_N_step(i,cfg,fringe_image):
    img1_group0 = []
    img1_group1 = []
    img1_group2 = []
    if i == 1:
        name = 0
    if i == 2:
        name = 1
    print(name)
    img1_0, img1_3, img1_6 = fringe_image[name][8], fringe_image[name][2], fringe_image[name][5]
    img1_group0.append(img1_0)
    img1_group0.append(img1_3)
    img1_group0.append(img1_6)
    img1_1, img1_4, img1_7 = fringe_image[name][0], fringe_image[name][3], fringe_image[name][6]
    img1_group1.append(img1_1)
    img1_group1.append(img1_4)
    img1_group1.append(img1_7)
    img1_2, img1_5, img1_8 = fringe_image[name][1], fringe_image[name][4], fringe_image[name][7]
    img1_group2.append(img1_2)
    img1_group2.append(img1_5)
    img1_group2.append(img1_8)

    phase_measures0 = PE(cfg).psi_extract(images=img1_group0)  # 用origin求包裹相位
    phase_measures1 = PE(cfg).psi_extract(images=img1_group1)  # 用2pai/3N求包裹相位
    phase_measures2 = PE(cfg).psi_extract(images=img1_group2)  # 用4pai/3N求包裹相位

    return phase_measures0,phase_measures1,phase_measures2,img1_group0,img1_group1,img1_group2

class Recons3D():
    def __init__(self, cfg):
        self._c = cfg
        self.calibrator = Calibrator(self._c)
        self.map_c1, self.map_c2, self.phase_rectified, self.Q = self.calibrator.load()

        self.phase_rectified = self.phase_rectified / self._c.Tc[0]

        self.pe = eval(self._c.pe_method)(self._c)

        self.shape = self.phase_rectified.shape

    def measure(self, images, gray,aa=0):
        if aa == 0:
            print("wrapped phase")
            phase = self.pe.psi_extract(images[0])
            # phase, _ = self.pe.phase_extract(images)
            return phase

        if aa == 1:
            print("unwrapped phase")
            phase, _ = self.pe.basic_extract2(images)

            phase1 = cv2.remap(phase, self.map_c1, self.map_c2, cv2.INTER_LINEAR)

            gray = cv2.remap(gray, self.map_c1, self.map_c2, cv2.INTER_CUBIC)
            gray_mask = gray < 35
            # print("gray_mask", gray_mask)

            s_x, s_y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
            s_x, s_y = s_x.astype(np.float32), s_y.astype(np.float32)
            x = np.zeros_like(self.phase_rectified).astype(np.float32)

            alpha = 0.9
            for i in range(10):
                alpha *= 0.9
                wp = cv2.remap(self.phase_rectified, s_x - x, s_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                diff = wp - phase1
                if i < 5:
                    diff = cv2.blur(diff, (3, 3))
                x = x + alpha * diff.astype(np.float32)
            disp = x
            mask = disp < -600
            disp[mask] = np.nan
            mask = disp > 700
            disp[mask] = np.nan
            # disp[gray_mask] = np.nan

            points = cv2.reprojectImageTo3D(disp.astype(np.float32), self.Q)
                # self.points = self.points[45:-35, 40:-5,:]

            if self._c.debug:
                plt.figure(figsize=(16, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(phase1)
                plt.colorbar()
                plt.title("phase measured")
                plt.subplot(1, 2, 2)
                plt.imshow(self.phase_rectified)
                plt.colorbar()
                plt.title("phase rectified")

                plt.figure(figsize=(16, 6))
                plt.imshow(disp)
                plt.colorbar()
                plt.title("disparity")
            return phase, points
        if aa == 2:
            old_psi = []
            p_m0, p_m1, p_m2, image_pai_2N_f1, image_oringe_f1, image_pai_N_f1 = old_Triple_N_step(
                1, self._c, images)
            correct_old_triple_n_l = phase_average(p_m0, p_m1, p_m2)
            phase_measures33, phase_measures44, phase_measures55, image_pai_2N_f2, image_oringe_f2, image_pai_N_f2 = old_Triple_N_step(
                2, self._c, images)
            correct_old_triple_n_h = phase_average(phase_measures33, phase_measures44, phase_measures55)
            old_psi.append(correct_old_triple_n_l)
            old_psi.append(correct_old_triple_n_h)

            old_Triple_N_step_unwrapped_phase, old_Triple_N_step_T = unwrapped_phase(self._c, old_psi)

            phase1 = cv2.remap(old_Triple_N_step_unwrapped_phase, self.map_c1, self.map_c2, cv2.INTER_LINEAR)

            gray = cv2.remap(gray, self.map_c1, self.map_c2, cv2.INTER_CUBIC)
            gray_mask = gray < 35
            # print("gray_mask", gray_mask)

            s_x, s_y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
            s_x, s_y = s_x.astype(np.float32), s_y.astype(np.float32)
            x = np.zeros_like(self.phase_rectified).astype(np.float32)

            alpha = 0.9
            for i in range(10):
                alpha *= 0.9
                wp = cv2.remap(self.phase_rectified, s_x - x, s_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                diff = wp - phase1
                if i < 5:
                    diff = cv2.blur(diff, (3, 3))
                x = x + alpha * diff.astype(np.float32)
            disp = x
            mask = disp < -600
            disp[mask] = np.nan
            mask = disp > 700
            disp[mask] = np.nan

            points = cv2.reprojectImageTo3D(disp.astype(np.float32), self.Q)
            return p_m0, p_m1, p_m2,correct_old_triple_n_l, old_Triple_N_step_unwrapped_phase, points
        if aa == 3:
            new_psi = []
            # 新的triple-N方法
            p_m0, p_m1, p_m2, image_4pai_3N_f1, image_oringe_f1, image_2pai_3N_f1 = new_Triple_N_step(
                1, self._c, images)
            correct_new_triple_n_l = phase_average(p_m0, p_m1, p_m2)
            phase_measures66, phase_measures77, phase_measures88, image_4pai_3N_f2, image_oringe_f2, image_2pai_3N_f2 = new_Triple_N_step(
                2, self._c, images)
            correct_new_triple_n_h = phase_average(phase_measures66, phase_measures77, phase_measures88)
            new_psi.append(correct_new_triple_n_l)
            new_psi.append(correct_new_triple_n_h)
            new_triple_n_step_unwrapped_phase, new_triple_n_step_t = unwrapped_phase(self._c, new_psi)

            phase1 = cv2.remap(new_triple_n_step_unwrapped_phase, self.map_c1, self.map_c2, cv2.INTER_LINEAR)

            gray = cv2.remap(gray, self.map_c1, self.map_c2, cv2.INTER_CUBIC)
            gray_mask = gray < 35
            # print("gray_mask", gray_mask)

            s_x, s_y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
            s_x, s_y = s_x.astype(np.float32), s_y.astype(np.float32)
            x = np.zeros_like(self.phase_rectified).astype(np.float32)

            alpha = 0.9
            for i in range(10):
                alpha *= 0.9
                wp = cv2.remap(self.phase_rectified, s_x - x, s_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                diff = wp - phase1
                if i < 5:
                    diff = cv2.blur(diff, (3, 3))
                x = x + alpha * diff.astype(np.float32)
            disp = x
            mask = disp < -600
            disp[mask] = np.nan
            mask = disp > 700
            disp[mask] = np.nan

            points = cv2.reprojectImageTo3D(disp.astype(np.float32), self.Q)

            return p_m0, p_m1, p_m2,correct_new_triple_n_l, new_triple_n_step_unwrapped_phase, points

    def remap(self, img):
        return cv2.remap(img, self.map_c1, self.map_c2, cv2.INTER_CUBIC)

    def save_points(self, fn, points=None):
        # if points is not None
        # verts = points
        # else: 
        verts = self.points if points is None else points
        # verts = self.points.reshape(-1, 3)
        verts = verts.reshape(-1, 3)
        mask = ~np.isnan(np.sum(verts, axis=-1))
        verts = verts[mask]
        # verts[np.isnan(verts)] = 0.00
        with open(fn, 'wb') as f:
            f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
            np.savetxt(f, verts, fmt='%f %f %f')