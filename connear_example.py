# -*- coding: utf-8 -*-
"""
This example script simulates the outputs of CoNNear, a CNN-based model of the 
auditory periphery, in response to a pure-tone stimulus for 3 different levels.
The stimulus can be adjusted to simulate different frequencies, levels or 
different types of stimulation (clicks, SAM tones).

CoNNear is consisted of three distinct modules, corresponding to the cochlear
stage, the IHC stage and the AN stage of the auditory periphery. The sampling
frequency of all CoNNear modules is 20 kHz. The CoNNear cochlea model can be 
substituted by one of the pre-trained hearing-impaired models to simulate the 
responses of a periphery with cochlear gain loss. Otoacoustic emissions (OAEs) 
can also be simulated from the cochlear outputs. Cochlear synaptopathy can also 
be simulated in the AN stage by adapting the number of AN fibers. 

To simulate population responses across CF after the AN level, the CN-IC python
module by Verhulstetal2018 (v1.2) is used as backend. The response waves I, 
III and V can be extracted to simulate auditory brainstem responses (ABRs) or 
envelope following responses (EFRs). 

The execution of CoNNear in Tensorflow can be significantly sped up with the 
use of a GPU.

@author: Fotios Drakopoulos, UGent, Feb 2022
"""

from connear_functions import *
import matplotlib.pyplot as plt
import ic_cn2018 as nuclei
from time import time

tf.compat.v1.disable_eager_execution() # speeds up execution in Tensorflow v2

#################### Simulation parameter definition #########################
# Define the pure tone stimulus
f_tone = 1e3 # frequency of the pure tone
L = np.array([30.,70.,90.]) # levels in dB SPL to simulate
stim_dur = 400e-3 # duration of the stimulus
# 设置了在刺激开始前的静默期为5毫秒。这个静默期可以帮助模拟真实世界中声音出现之前的静默背景。
# 在声音处理或分析中,声音突然出现可能导致一些非线性效应或处理上的困难（例如,在进行傅里叶变换时引入边缘效应）。
# 静默期提供了一个平稳的过渡,有助于减少这些潜在问题。
# 在实验设计中,静默期也是一种控制手段。
# 通过在不同的声音刺激之间插入静默期,可以确保每次刺激之间的独立性,避免刺激间的交互影响,从而保证实验结果的准确性和可重复性。
initial_silence = 5e-3 # silence before the onset of the stimulus
# 汉宁窗：这意味着在声音刺激的起始和结束部分,都会应用一个5毫秒长度的汉宁窗来平滑这些部分,从而减少这些区域的突变。
# 基于信号的采样频率（在这个例子中是fs = 20e3,即20kHz）和窗口的持续时间（5ms）,计算出汉宁窗的样本点数。
# 在20kHz的采样率下,5ms对应于20,000×0.005=100个样本点。接着,生成一个长度为100的汉宁窗系数数组。对于这100个样本点,按照汉宁窗的数学表达式计算每个点的窗函数值
# 将这个汉宁窗系数乘以声音信号的起始和结束的100个样本。这样,声音信号在这5毫秒内平滑地从0增加到其原始值,并在结束时从其原始值平滑降低到0。
# 有助于避免在分析时产生不必要的频谱泄露或其他人工效应
win_dur = 5.0e-3 # 5ms long hanning window for gradual onset
# 这是将声音刺激校准到的参考声压,单位为帕斯卡（Pascal）。2e-5 Pascal是国际上公认的听阈参考值,相当于最小可听声压级。
p0 = 2e-5 # calibrate to 2e-5 Pascal
# fmod = 98 # modulation frequency - uncomment for SAM tone stimuli
# m = 0.85  # modulation depth - uncomment for SAM tone stimuli

# Change the poles variable to include HI in the cochlear stage
poles = '' # choose between NH, Flat25, Flat35, Slope25, Slope35
# Simulate population response waves I, III, V & EFR
# 在听觉科学领域,人群响应通常用于描述听觉路径中不同级别的神经元群体对声音刺激的响应。这些响应可以通过各种生理测量方法记录,例如：
# 听脑干反应（ABR, Auditory Brainstem Responses）：是一种通过电极记录头皮上神经电活动来测量的反应,反映了听觉路径从耳蜗神经到脑干的集体电生理活动。
# ABR包括多个波形,其中波I、III和V最常用于临床和研究中,分别对应于听神经、下丘脑和中脑水平的人群响应。
# 包络跟随反应（EFR, Envelope Following Responses）：是对复杂声音（如调制声或语音）的包络特征的神经同步反应。
# EFR可以提供关于听觉系统如何处理声音的时域信息的重要信息。
# 通过分析这些人群响应,研究人员可以深入理解听觉系统的功能状态,诊断听力损失的性质和程度,以及评估听力康复措施（如助听器或耳蜗植入）的效果。
# 人群响应的分析也有助于研究听觉系统如何对不同类型的声音刺激进行编码,以及在不同听力条件下这些编码过程如何变化。
population_sim = 1 # set to 0 to omit the CN-IC stage
# Change the number of [HSR, MSR, LSR] ANFs to include cochlear synaptopathy for the population responses
# 突触病设置
num_anfs = [13,3,3] # 13,3,3 is the NH innervation, change to lower values (e.g. 13,0,0 or 10,0,0) to simulate cochlear synaptopathy
# Pick a channel number for which to plot the single-unit responses
# 指在模拟过程中对应于不同中心频率（CF）的听觉神经纤维或神经元群体。
No = 122 # between 0 and 200 (201 CFs are used for CoNNear) - 122 corresponds to ~1 kHz
# Simulate otoacoustic emissions (increases computation time)
# 耳声发射
oae_sim = 0 # set to 1 to generate and plot OAEs from the cochlear output

# number of parallely executed CFs for the IHC and ANF models
# 表示有201个不同的中心频率将在IHC和ANF模型中同时被模拟处理
# 用来模拟人耳内部听神经纤维对声音刺激反应的计算模型。
# 神经纤维（Auditory Nerve Fiber）模型用来模拟人耳内部听神经纤维对声音刺激反应的计算模型。
# 听神经纤维是连接内毛细胞（Inner Hair Cells, IHCs）和大脑听觉中枢的神经元,负责将声音信息从耳蜗传递到大脑。
Ncf = 201 # set to 1 if the execution is too slow

#################### Main script #############################################
# Model-specific variables
# 这些参数是为了在模拟过程中确保ANF模型能够充分考虑到声音信号的时间上下文信息,以模拟人耳对声音信号处理的真实情况。
# 上下文信息对于模型捕捉到声音信号的动态变化特别重要,可以帮助模型更准确地模拟听觉系统的响应。
fs = 20e3
context_left = 7936 # samples (396.8 ms) - left context of the ANF model
context_right = 256 # samples (12.8 ms) - right context 
# load model CFs 
CF = np.loadtxt('CoNNear_periphery/connear/cf.txt')*1e3
# scaling applied to the CoNNear outputs to bring them back to the original representations
cochlea_scaling = 1e6  # 耳蜗模型输出的缩放因子是100万（1e6）。应用这个缩放因子是为了将耳蜗阶段的输出缩放回其原始的生理学表示。
ihc_scaling = 1e1  # 内毛细胞阶段的输出需要乘以10（1e1）来缩放回原始表示。这个较小的缩放因子反映了从耳蜗到IHC阶段输出变化的生理学范围可能不像耳蜗阶段那么大。
an_scaling = 1e-2  # 对于听神经阶段的输出,缩放因子是0.01（1e-2）。
# CoNNear model directory
modeldir = 'CoNNear_periphery/connear/'
# number of layers in the deepest architecture used (ANF model) - for padding the input accordingly
# 这个层数信息可能被用来决定如何对模型的输入数据进行预处理,特别是在需要将输入数据的大小调整到模型能够接受的特定形状时。
# 在深度学习模型中,特别是在处理序列数据（如音频信号）时,常常需要通过添加额外的数据（通常是0）来扩展输入序列的长度,以确保它们符合模型要求的维度。
Nenc = 14
# OAE parameters - shifted windows of the stimulus are used to get a smooth frequency representation
# 配置了用于模拟耳声发射的参数
oae_cf_no = 0 # use the highest frequency channel (~12 kHz)  # 实际的OAE测量中,不同频率的耳声发射反映了耳蜗不同部位的健康状况。
# 这定义了用于分析的窗口大小为4096个样本。在处理音频或信号时,窗口函数用于局部分析信号,这里的窗口大小决定了分析的频率分辨率和时间分辨率。
# 较大的窗口提供了更好的频率分辨率,但较差的时间分辨率。
# 第一个窗口是1-4096,那么由于步长是50,滴第二个窗口就是51-4146
oae_window = 4096 # use a smaller window to generate shifted slices of the full stimulus
# 指定了窗口移动的步长为50个样本。步长决定了窗口是如何沿着信号移动的,较小的步长会生成更多的窗口和更平滑的频率响应,但同时也会增加计算量和模拟时间。
# 减小步长意味着窗口移动时跳过更少的样本点,从而产生更多的重叠窗口。
# 虽然减小步长可以提高分析的精度和平滑度,但这也意味着需要处理更多的窗口,增加处理时间
oae_step = 50 # the step with which the window is shifted - decrease to get smoother responses (longer simulation times)
# 这表示耳蜗模型在处理每个窗口的输入时需要额外的256个样本作为上下文。在信号处理中,上下文用于提供窗口之外的额外信息,以改善边缘处理和过渡效应。
cochlea_context = 256 # the cochlear model requires 256 samples of context on each side of the input

# Make stimulus
# 生成了一个时间序列,从0秒开始,到stim_dur秒结束,时间间隔为1/fs秒（即采样间隔）,用于后续信号的生成。
t = np.arange(0., stim_dur, 1./fs)
# 这行代码生成一个所有值都为1的数组,模拟了点击声。点击声是一种短暂且强度相同的声音刺激
#stim_sin = np.ones(t.shape) # uncomment for click stimuli
stim_sin = np.sin(2 * np.pi * f_tone * t) # uncomment for pure-tone stimuli
# 调幅音,其中m表示调制深度,fmod表示调制频率。调幅音由一个载波（纯音）和一个调制波（余弦波）组成,通过改变调制深度和调制频率,可以生成不同特性的声音刺激。
#stim_sin = (1 + m * np.cos(2 * np.pi * fmod * t)) * np.sin(2 * np.pi * f_tone * t) # uncomment for SAM tone stimuli
# apply hanning window
if win_dur:
    # 计算出汉宁窗的总长度,将覆盖起始的100个样本和结束的100个样本。
    winlength = int(2*win_dur * fs)
    # 这行代码调用SciPy库的windows.hann函数生成长度为winlength的汉宁窗系数。这些系数用于平滑地调制信号的起始和结束部分。
    win = sp_sig.windows.hann(winlength) # double-sided hanning window
    # 分别指向信号的开始的100个样本和结束的一半汉宁窗长度部分,也即100个样本。
    # 通过与汉宁窗系数相乘,信号的这两部分被平滑地调制。
    stim_sin[:int(winlength/2)] = stim_sin[:int(winlength/2)] * win[:int(winlength/2)]
    stim_sin[-int(winlength/2):] = stim_sin[-int(winlength/2):] * win[int(winlength/2):]
total_length = context_left + int(initial_silence * fs) + len(stim_sin) + context_right
# 这行代码初始化一个二维信号数组stim,其形状为len(L)行和total_length列,填充值为0。
# len(L)代表不同声级的数量,即对于每个声级都将生成一个刺激信号。
stim = np.zeros((len(L), total_length))
# 这行代码计算了在stim数组中,纯音信号应该插入的列的范围。
stimrange = range(context_left + int(initial_silence * fs), context_left + int(initial_silence * fs) + len(stim_sin))
# RMS用于衡量声音的平均功率或响度。因为RMS考虑了信号在整个周期内的所有值。这使得RMS成为评估信号持续效应（如响度、热量产生或电力消耗）的理想选择。
# 「p0 * 10**(L[i]/20)」计算绝对声压级,这意味着,为了达到L[i] dB SPL,信号的绝对声压级应该调整到「p0 * 10**(L[i]/20)」 Pa。
# rms(stim_sin)计算声音信号的RMS值, stim_sin / rms(stim_sin)进行归一化
# 我们通过乘以绝对声压级「p0 * 10**(L[i]/20)」来调整归一化信号的声压级到目标级别 L[i]
for i in range(len(L)):
    stim[i, stimrange] = p0 * 10**(L[i]/20) * stim_sin / rms(stim_sin) # calibrate
'''增加一个纬度。
原始：(3, 10)
[[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
 [1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2. ]
 [2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3. ]]
增加一个纬度(3, 10, 1)
[[[0.1]
  [0.2]
  [0.3]
  [0.4]
  [0.5]
  [0.6]
  [0.7]
  [0.8]
  [0.9]
  [1. ]]
  
  ...

  ]]
 '''
stim = np.expand_dims(stim, axis=2) # make the stimulus 3D

# Check the stimulus time-dimension size
# stim.shape[2]是时间维度也是信号长度
if stim.shape[1] % 2**Nenc: # input size needs to be a multiple of 16384 for the ANF model
    # 「int(np.ceil(stim.shape[1]/(2**Nenc)))」计算出当前时间维度的长度除以2^14(层数)的向上取整结果
    Npad = int(np.ceil(stim.shape[1]/(2**Nenc)))*(2**Nenc)-stim.shape[1]
    # 「(0,Npad)」意思是在第二个纬度上前面添加0个0,后面添加Npad个0
    # 具体选择16384作为倍数的原因取决于模型的设计。在深度学习中,选择作为数据维度的大小可以优化某些计算库（如FFT算法、GPU加速库等）的性能。
    # 在这个案例中,2^14=16384可能是基于模型结构（如深度、层类型、数据处理策略）和计算效率的考量。
    stim = np.pad(stim,((0,0),(0,Npad),(0,0))) # zero-pad the 2nd dimension (time)

# Load each CoNNear module separately
cochlea, ihc, anf = build_connear(modeldir,poles=poles,Ncf=Ncf) # load the CoNNear models

print('CoNNear: Simulating auditory periphery stages')
# 开始计时以测量执行接下来代码部分所需时间。
time_elapsed=time()

# Cochlea stage
# vbm保存了耳蜗模型的输出, 即基底膜的振动响应。
vbm = cochlea.predict(stim) # BM vibration
# IHC-ANF stage
# 接下来的代码根据中心频率数(Ncf)的不同，选择不同的模型来模拟内毛细胞(IHC)和听神经纤维(ANF)的响应。
if Ncf == 201: # use the 201-CF models
    vihc = ihc.predict(vbm) # IHC receptor potential
    ranf = anf.predict(vihc) # ANF firing rate
else: # use the 1-CF IHC-ANF models
    vihc = np.zeros(vbm.shape)
    ranf = np.zeros((3,vihc.shape[0],vihc.shape[1]-context_left-context_right,vihc.shape[2]))
    for cfi in range (0,CF.size): # simulate one CF at a time to avoid memory issues
        vihc[:,:,[cfi]] = ihc.predict(vbm[:,:,[cfi]])
        ranf[:,:,:,[cfi]] = anf.predict(vihc[:,:,[cfi]])

time_elapsed = time() - time_elapsed
print('Simulation finished in ' + '%.2f' % time_elapsed + ' seconds')

# Simulate otoacoustic emissions
if oae_sim:
    print('CoNNear: Simulating otoacoustic emissions')
    oae_size = oae_window # size of the oae response to keep 
    oae_min_window = oae_window-oae_step-int((initial_silence+win_dur)*fs) # minimum length of slice to include 
    oae = np.zeros((vbm.shape[0],oae_size,vbm.shape[2])) # pre-allocate array
    for li in range(0,L.size):
        # produce shifted versions of the input signal to get smoother OAEs
        # 在模拟OAEs时移除上下文信息，是为了确保分析的准确性、减少不必要的计算负担，并专注于信号的有效和关键部分。
        stim_oae = stim[li,context_left:-context_right,0] # use the stimulus without context # 为什么要去除上下文？
        # 将stim_oae分割成多个重叠的小片段（stim_oae_slices），这通过移动窗口（由oae_window和oae_step确定）实现，以获得更平滑的OAE响应。
        stim_oae_slices = slice_1dsignal(stim_oae, oae_window, oae_step, oae_min_window, left_context=cochlea_context, right_context=cochlea_context) # 256 samples of context are added on the sides
        vbm_oae_slices = cochlea.predict(stim_oae_slices) # simulate the outputs for the generated windows
        # 从vbm_oae_slices中移除添加的上下文边界，保留耳蜗响应的核心部分。
        vbm_oae_slices = vbm_oae_slices[:,cochlea_context:-cochlea_context,:] # remove the context from the cochlear outputs
        # undo the windowing to get back the full response
        vbm_oae = unslice_3dsignal(vbm_oae_slices, oae_window, oae_step, fs=fs, trailing_silence=initial_silence+win_dur) # use the steady-state response for the fft (omit silence and onset)
        # 根据cochlea_scaling进行调整，以反映真实的耳蜗放大效应。
        oae[li,:,:] = vbm_oae[:,:oae_size,:] / cochlea_scaling
    oae_fft, oae_nfft = compute_oae(oae, cf_no=oae_cf_no) # compute the fft of the oae response

# Rearrange the outputs, omit context and scale back to the original values
stim = stim[:,context_left:-context_right,:] # remove context from stim
# 信号在模拟过程中可能经过了一定的放大，以适配模型的输入输出要求或提高处理的精确度。现在，需要将它们缩放回原始的比例。
vbm = vbm[:,context_left:-context_right,:] / cochlea_scaling # omit context from the uncropped outputs
vihc = vihc[:,context_left:-context_right,:] / ihc_scaling
ranf_hsr = ranf[0] / an_scaling
ranf_msr = ranf[1] / an_scaling
ranf_lsr = ranf[2] / an_scaling
del ranf

# Simulate CN and IC stages
# 耳蜗核（Cochlear Nuclei, CN）
# 下丘脑（Inferior Colliculus, IC）
# 听觉脑干反应（Auditory Brainstem Response, ABR）中的波形I、III和V，这些波形是评估听觉系统功能的重要指标。
# 包络跟随反应（Envelope Following Responses, EFR），这是对声音信号包络变化的神经反应。
if population_sim:
    print('Simulating IC-CN stages (Verhulstetal2018 v1.2)')
    # the CN/IC stage of the Verhulstetal2018 model (v1.2) is used
    # 为每个声级(L.size)，初始化用于存储CN和IC输出以及ANF响应总和的数组。
    cn = np.zeros(ranf_hsr.shape)
    an_summed = np.zeros(ranf_hsr.shape)
    ic = np.zeros(ranf_hsr.shape)
    for li in range(0,L.size):
        # nuclei.cochlearNuclei ---> ic_cn2018.py
        # 计算CN阶段的输出和ANF响应总和
        cn[li,:,:],an_summed[li,:,:]=nuclei.cochlearNuclei(ranf_hsr[li],ranf_msr[li],ranf_lsr[li],num_anfs[0],num_anfs[1],num_anfs[2],fs)
        # 计算IC阶段的输出
        ic[li,:,:]=nuclei.inferiorColliculus(cn[li,:,:],fs)
    # compute response waves 1, 3 and 5
    w1=nuclei.M1*np.sum(an_summed,axis=2)
    w3=nuclei.M3*np.sum(cn,axis=2)
    w5=nuclei.M5*np.sum(ic,axis=2)
    # EFR is the summation of the W1 W3 and W5 responses
    EFR = w1 + w3 + w5
    
    # EFR spectrum
    # 展示了如何计算EFR信号的频谱。这通过对EFR信号执行快速傅里叶变换（FFT）并计算其幅度来实现。这个频谱分析有助于理解EFR对不同频率成分的响应
    #EFR_sig = EFR[:,int((initial_silence+win_dur)*fs):int((initial_silence+stim_dur)*fs)] # keep only the signal part
    #nfft = next_power_of_2(EFR_sig.shape[1]) # size of fft
    #EFR_fft = np.fft.fft(EFR_sig,n=nfft) / EFR_sig.shape[1] # compute the fft over the signal part and divide by the length of the signal
    #nfft = int(nfft/2+1) # keep one side of the fft
    #EFR_fft_mag = 2*np.absolute(EFR_fft[:,:nfft])
    #freq = np.linspace(0, fs/2, num = nfft)
    
#################### Plot the responses ######################################
t = np.arange(0., ranf_hsr.shape[1]/fs, 1./fs)
ranf_hsr_no = ranf_hsr[:,:,No].T
ranf_msr_no = ranf_msr[:,:,No].T
ranf_lsr_no = ranf_lsr[:,:,No].T

# v_bm and V_IHC results
vbm_rms = rms(vbm, axis=1).T
ihc_rms = np.mean(vihc, axis=1).T
vbm_no = vbm[:,:,No].T
vihc_no = vihc[:,:,No].T

if oae_sim:
    oae_no = oae[:,:,oae_cf_no].T
    oae_freq = np.linspace(0, fs/2, num = oae_nfft)
    
    plt.figure(1, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(2,1,1),plt.plot(1000*t[:oae_size],oae_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.ylabel('Ear Canal Pressure [Pa]'),plt.xlabel('Time [ms]')
    plt.title('CF of ' + '%.2f' % CF[oae_cf_no] + ' Hz')
    plt.subplot(2,1,2),plt.plot(oae_freq/1000,20*np.log10(oae_fft.T/p0)),plt.grid()
    plt.ylabel('EC Magnitude [dB re p0]'),plt.xlabel('Frequency [kHz]'),plt.xlim(0,10)
    plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    plt.tight_layout()

plt.figure(2, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
plt.subplot(2,2,1),plt.plot(1000*t,1e6*vbm_no[:,::-1]),plt.grid()
plt.xlim(0,50),plt.ylabel('$v_{bm}$ [${\mu}m$/s]'),plt.xlabel('Time [ms]')
plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
plt.subplot(2,2,2),plt.plot(CF/1000,20*np.log10(1e6*vbm_rms[:,::-1])),plt.grid()
plt.ylabel('rms of $v_{bm}$ [dB re 1 ${\mu}m$/s]'),plt.xlabel('CF [kHz]'),plt.xlim(0,8)
plt.title('Excitation Pattern')
plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
plt.subplot(2,2,3),plt.plot(1000*t,1e3*vihc_no[:,::-1]),plt.grid()
plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('$V_{ihc}$ [mV]')
plt.subplot(2,2,4),plt.plot(CF/1000,1e3*ihc_rms[:,::-1]),plt.grid()
plt.xlabel('CF [kHz]'),plt.ylabel('rms of $V_{ihc}$ [mV]'),plt.xlim(0,8)
plt.tight_layout()

# single-unit responses
plt.figure(3, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
plt.subplot(3,2,1),plt.plot(1000*t,ranf_hsr_no[:,::-1]),plt.grid()
plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('HSR fiber [spikes/s]')
plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
plt.subplot(3,2,3),plt.plot(1000*t,ranf_msr_no[:,::-1]),plt.grid()
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('MSR fiber [spikes/s]')
plt.subplot(3,2,5),plt.plot(1000*t,ranf_lsr_no[:,::-1]),plt.grid()
plt.xlim(0,100),plt.xlabel('Time [ms]'),plt.ylabel('LSR fiber [spikes/s]')
plt.tight_layout()

if population_sim:
    an_summed_no = an_summed[:,:,No].T
    cn_no = cn[:,:,No].T
    ic_no = ic[:,:,No].T
    
    # single-unit responses
    plt.subplot(3,2,2),plt.plot(1000*t,an_summed_no[:,::-1]),plt.grid()
    plt.title('CF of ' + '%.2f' % CF[No] + ' Hz')
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('sum AN [spikes/s]')
    plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    # Spikes summed across all fibers @ 1 CF
    plt.subplot(3,2,4),plt.plot(1000*t,cn_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('CN [spikes/s]')
    plt.subplot(3,2,6),plt.plot(1000*t,ic_no[:,::-1]),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('IC [spikes/s]')
    plt.tight_layout()

    # population responses
    plt.figure(4, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    plt.subplot(4,1,1),plt.plot(1000*t,1e6*w1[::-1].T),plt.grid()
    plt.title('Population Responses summed across simulated CFs')
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-1 [${\mu}V$]')
    #plt.legend(["%d" % x for x in L[::-1]],frameon=False,loc='upper right')
    plt.subplot(4,1,2),plt.plot(1000*t,1e6*w3[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-3 [${\mu}V$]')
    plt.subplot(4,1,3),plt.plot(1000*t,1e6*w5[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('W-5 [${\mu}V$]')
    plt.subplot(4,1,4),plt.plot(1000*t,1e6*EFR[::-1].T),plt.grid()
    plt.xlim(0,50),plt.xlabel('Time [ms]'),plt.ylabel('EFR [${\mu}V$]')
    plt.tight_layout()
    
    # EFR spectrum
    #plt.figure(5, figsize=(10, 6), dpi=300, facecolor='w', edgecolor='k')
    #plt.plot(freq,EFR_fft_mag.T*1e6),plt.grid()
    #plt.title('EFR frequency spectrum')
    #plt.xlim(0,10000),plt.xlabel('Frequency [Hz]'),plt.ylabel('EFR Magnitude [${\mu}V$]')
    #plt.tight_layout()

plt.show()
