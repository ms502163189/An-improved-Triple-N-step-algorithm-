# An-improved-Triple-N-step-algorithm
# 文件说明 

 - data文件夹
 该文件夹内是所有的实验数据
 ./data/calibration/2/  保存了标定后的文件
 ./data/patterns/         保存了生成的条纹图
 ./data/recordings/2/  保存了照相机拍摄的被测物体条纹图
 ./data/results/             保存了Triple-N-steps-real.ipynb实际测试的结果图
 
 - calibration.py
 标定投影仪和照相机使用，标定的结果会保存在./data/calibrations/2/calibration_result.npz文件内
 
 - camera.py
照相机设置文件，主要在capture.py中使用
 
 - capture.py
 照相机拍照时使用，投影仪投射20步、12步、9步、3步相移条纹投影在物体，照相机拍摄图片。拍摄的图片分别保存在./data/recordings/2/20-step、./.../12-step、./.../9-step、./.../3-step内中
 
 - config.py
 配置文件，里面包含条纹的基本参数：频率、相移步数、保存文件等等
 
 - fringes.py
 生成条纹，在这里主要是分别生成20、12、9、3步相移条纹
 - phase.py
主要用来解相位，可以得到包裹相位以及解包裹相位。
 - projector.py
投影仪的设置文件，在capture.py中调用
 - recons3d.py
用于实际测试的3维重建，完成相位到点的变换
 - Triple-N steps.ipynb
仿真测试提出的Triple-N steps相移算法的效果
 - Triple-N-steps-real.ipynb
实际测试提出的Triple-N steps相移算法的效果，结果还会保存在./data/results/内

# 实验流程：
# *标定过程：*
 - 首先用三频四步来进行标定，修改config.py里面的配置
 
 - 接着运行fringes.py生成相应的电子版条纹图，包括一张背景
 
 - 接着打开capture.py，下图所示，标定和实验要用不同的函数，记得改注释

 - 相机捕获到图片后，一组棋盘格会生成13张图片，12张条纹图，还有一个棋盘格背景图用于检测角点
 
 - 之后运行calibration.py即可，其标定结果会保存在./data/2/calibration_result.npz
 
 - 最后会存储极线矫正和畸变矫正的一些参数，深度差异矩阵（三维重建），相位预处理矩阵
 # *实验过程：*

 - 首先修改config.py里面的配置，均在两频下完成，修改频率值

 - 接着运行fringes.py生成相应的20步、12步、9步、3步电子版条纹图及背景图

 - 接着打开capture.py，标定和实验要用不同的函数，记得改注释（文件内已经用注释标记清楚了）
 
 - 相机捕获到图片后，分别保存在./data/recordings/2/20-step、./.../12-step、./.../9-step、./.../3-step内中

 - 之后运行Triple-N-steps-real.ipynb即可查看实际测试的运行效果，可以对比看到我们改进的Triple-N-steps方法与传统的Triple-N-steps以及9步相移法、三步相移法解相位及三维重建的精度。

 - 运行Triple-N-steps.ipynb可以查看理论仿真运行结果，仿真运行效果不需要上述过程，直接运行即可看到效果。


