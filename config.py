import numpy as np

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def config():
    """default configure"""
    cfg= AttrDict()
    cfg.debug = True

    # **********************************************************************************
    # The fringe pattern generation ****************************************************
    cfg.pattern_size = [1920, 1080]
    cfg.camera_size = [1920, 1080]
    cfg.pattern_path = './data/patterns'

    #--- The sine wave projection patten  I(x)=A + B*sin(\pi x/T+\phi_0)
    cfg.scale = 3.0 # the projector resolution divides the camera resolution
    # cfg.Tc = [10, 11, 12]  # calibration             
    # cfg.Tc = [14, 15, 16]  # measurement
    # cfg.Tp = [cfg.scale * T for T in cfg.Tc]  # cfg.scale*cfg.Tc
    cfg.Tc = [170, 190, 210]  # measurement
    cfg.Tp = [170, 190, 210]  # measurement
    # print("Tp", cfg.Tp)
    # cfg.Tp = [200, 210, 220]  # 在这个实验中，就先设置成这个周期。
    cfg.A = [128,128,128]
    cfg.B = [96,96,96]
    step  = 3
    cfg.steps = [step,step,step]
    cfg.light_value = 80
    cfg.hv = "h" # "h" "v"---horizontal or vertical


    cfg.alpha = [1, cfg.Tp[0]/cfg.Tp[1], cfg.Tp[0]/cfg.Tp[2]]
    cfg.gamma = 1.2
    cfg.C, cfg.D, cfg.E , cfg.F= 0,0,0,0

    # **********************************************************************************
    # The images recording *************************************************************
    cfg.img_sz= (640,480) # I don't have a high resolution industrial camera at hand
    cfg.calibra_path = './data/calibrations/2'
    cfg.measure_path = './data/recordings/2/'
    
    # **********************************************************************************
    # phase extraction method **********************************************************
    cfg.pe_method = "PE"  # "PE", "LLS", "MPE", "CFPE"
    
    W, H = cfg.pattern_size
    cfg.y = np.linspace(0,H-1,H).reshape(-1,1) if cfg.hv == "h" else np.linspace(0,W-1,W)
    cfg.coefficient = np.zeros([int(len(cfg.y)/40),6])
#     cfg.x = np.linspace(0,W-1,W) if cfg.hv == "h" else np.linspace(0,H-1,H).reshape(-1,1)
#     cfg.coefficient = np.zeros([int(len(cfg.x)/10),6])
    cfg.net_dir="./data/train3/"
    cfg.net_after_dir="./data/train3_after/"
    cfg.model_dir = "./data/model3/"
    cfg.tensorboard_dir = "./path/to/log/"
    cfg.copy_dir = "./data/best_data_copy_dir/"

#     # The image processing
#     cfg.GCME = True # "None", ""

    cfg.MaxIter = 6
    return cfg

# def config():
#     """default configure"""
#     cfg= AttrDict()
#     cfg.debug = True
    
#     # **********************************************************************************
#     # The images generation ************************************************************
#     cfg.pattern_sz= (1080,1920)
#     cfg.pattern_path = './data/patterns'

#     #- The sine wave projection patten  I(x)=A + B*sin(\pi x/T+\phi_0)
#     cfg.steps = 6 
#     cfg.scale = 3.0 # the projector resolution divides the camera resolution
#     # camera config, 
#     # cfg.Tc1,cfg.Tc2,cfg.Tc3 = 10, 11, 12 
#     cfg.Tc1,cfg.Tc2,cfg.Tc3 = 16, 18, 20 
#     # projector config
#     cfg.Tp1, cfg.Tp2, cfg.Tp3 = cfg.Tc1*cfg.scale, cfg.Tc2*cfg.scale, cfg.Tc3*cfg.scale
#     cfg.A1,cfg.A2,cfg.A3 = 128,128,128
#     cfg.B1,cfg.B2,cfg.B3 = 96, 96, 96


#     # #- The vertical sine wave for calibration I(y) = A+B*sin(\pi y/T +\phi_0
#     # cfg.T1_y,cfg.T2_y,cfg.T3_y =  12, 13, 14 #24., 27., 30.#13, 16, 20.5 # 24.00, T2:27.00,   T3:30.00
#     # cfg.A1_y,cfg.A2_y,cfg.A3_y = 128,128,128
#     # cfg.B1_y,cfg.B2_y,cfg.B3_y =96,96,96

#     #- A constant light value for capturing the chessboard 
#     cfg.light_value = 250

    
#     # **********************************************************************************
#     # The images recording *************************************************************
#     cfg.img_sz= (640,480) # I don't have a high resolution industrial camera at hand
#     cfg.cali_path = './data/calibrations/1'
#     cfg.measure_path = f"./data/recordings/s{cfg.steps}"


#     # The image processing
#     # cfg.gamma = "GCME" # "None", ""

#     return cfg

