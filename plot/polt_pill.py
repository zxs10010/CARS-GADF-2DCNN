import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pyts.image import MarkovTransitionField, GramianAngularField
from Getdata_grain_casein_pill import getdata_Grain_casein


plt.rc('font', family='Times New Roman')
# Plot the time series and its recurrence plot
width_ratios = (7, 7,7,7, 0.4)
height_ratios = (7,)
width = 20
height = width * sum(height_ratios) / sum(width_ratios)
fig = plt.figure(figsize=(width, height))
gs = fig.add_gridspec(1, 5,
                      width_ratios=width_ratios,
                      height_ratios=height_ratios,
                      left=0.04, right=0.97, bottom=0.15, top=0.98,
                      wspace=0.15, hspace=0
                      )

select_method='crs'
cc=6
test_data, test_lable, train_data, train_lable, n_components, a, b = getdata_Grain_casein(fs=select_method,c=cc)
train_data = train_data.reshape(a, 1, n_components)
train_data0 = train_data[0]
# print(train_data0)
mtf = GramianAngularField(image_size=n_components, method='difference')
train_data_mtf = mtf.fit_transform(train_data0)

ax_gasf = fig.add_subplot(gs[0, 0])
im=ax_gasf.imshow(train_data_mtf[0], cmap='rainbow', origin='upper',)
ax_gasf.set_xticks(np.arange(0,404,100))
ax_gasf.set_yticks(np.arange(0,404,100))
plt.yticks(fontproperties = 'Times New Roman')
plt.xticks(fontproperties = 'Times New Roman')
ax_gasf.set_title('(a)', y=-0.175,font={'family':'Times New Roman', 'size':30})
plt.tick_params(labelsize=30,width=3)
plt.tick_params(pad=0)
# print(train_data0)


select_method='cars'
cc=6
test_data, test_lable, train_data, train_lable, n_components, a, b = getdata_Grain_casein(fs=select_method,c=cc)
train_data = train_data.reshape(a, 1, n_components)
train_data0 = train_data[0]
mtf = GramianAngularField(image_size=n_components, method='difference')
train_data_mtf = mtf.fit_transform(train_data0)

ax_gasf = fig.add_subplot(gs[0, 3])
ax_gasf.imshow(train_data_mtf[0], cmap='rainbow', origin='upper',)
ax_gasf.set_xticks(np.arange(0,42,10))
ax_gasf.set_yticks(np.arange(0,42,10))
plt.yticks(fontproperties = 'Times New Roman')
plt.xticks(fontproperties = 'Times New Roman')
ax_gasf.set_title('(d)', y=-0.175,font={'family':'Times New Roman', 'size':30})
plt.tick_params(labelsize=30,width=3)
plt.tick_params(pad=0)

select_method='iriv'
cc=10
test_data, test_lable, train_data, train_lable, n_components, a, b = getdata_Grain_casein(fs=select_method,c=cc)
train_data = train_data.reshape(a, 1, n_components)
train_data0 = train_data[0]

mtf = GramianAngularField(image_size=n_components, method='difference')
train_data_mtf = mtf.fit_transform(train_data0)

ax_gasf = fig.add_subplot(gs[0, 2])
ax_gasf.imshow(train_data_mtf[0], cmap='rainbow', origin='upper',)
ax_gasf.set_xticks(np.arange(0,39,10))
ax_gasf.set_yticks(np.arange(0,39,10))
plt.yticks(fontproperties = 'Times New Roman')
plt.xticks(fontproperties = 'Times New Roman')
ax_gasf.set_title('(c)', y=-0.175,font={'family':'Times New Roman', 'size':30})
plt.tick_params(labelsize=30,width=3)
plt.tick_params(pad=0)

select_method='vcpa'
cc=1
test_data, test_lable, train_data, train_lable, n_components, a, b = getdata_Grain_casein(fs=select_method,c=cc)
train_data = train_data.reshape(a, 1, n_components)
train_data0 = train_data[0]

mtf = GramianAngularField(image_size=n_components, method='difference')
train_data_mtf = mtf.fit_transform(train_data0)

ax_gasf = fig.add_subplot(gs[0, 1])
ax_gasf.imshow(train_data_mtf[0], cmap='rainbow', origin='upper',)
ax_gasf.set_xticks(np.arange(0,16,5))
ax_gasf.set_yticks(np.arange(0,16,5))
plt.yticks(fontproperties = 'Times New Roman')
plt.xticks(fontproperties = 'Times New Roman')
ax_gasf.set_title('(b)', y=-0.175,font={'family':'Times New Roman', 'size':30})
plt.tick_params(labelsize=30,width=3)
plt.tick_params(pad=0)

l = 0.945
b = 0.15
w = 0.013
h = 0.98-b

#对应 l,b,w,h；设置colorbar位置；
rect = [l,b,w,h]
cbar_ax = fig.add_axes(rect)
cb = plt.colorbar(im, cax=cbar_ax,format='Times New Roman')
cb.ax.tick_params(labelsize=30)
ticks = (-1,-0.5,0,0.5,1)
ticklabels = (-1,-0.5,0,0.5,1)

cb.set_ticks(ticks)
cb.set_ticklabels(ticklabels)
# cb.ax.tick_params(fontproperties = 'Times New Roman')
# #设置colorbar标签字体等
plt.rc('font',family='Times New Roman')
matplotlib.rcParams['font.family'] = 'Times New Roman'
# font = {'family' : 'serif',
# #       'color'  : 'darkred',
#     'color'  : 'black',
#     'weight' : 'normal',
#     'size'   : 16,
#     }
# cb.set_label('T' ,fontdict=font) #设置colorbar的标签字体及其大小

# ax_cbar = fig.add_subplot(rect)
# fig.colorbar(im,cax=ax_cbar,)
# #对应 l,b,w,h；设置colorbar位置；
# rect = [l,b,w,h]
# cbar_ax = fig.add_axes(rect)
# cb = plt.colorbar(im, cax=cbar_ax)
# cb.ax.tick_params(labelsize=22)
# # cb.ax.tick_params(fontproperties = 'Times New Roman')
# # #设置colorbar标签字体等
# plt.rc('font',family='Times New Roman')
# matplotlib.rcParams['font.family'] = 'Times New Roman'
# # font = {'family' : 'serif',
# # #       'color'  : 'darkred',
# #     'color'  : 'black',
# #     'weight' : 'normal',
# #     'size'   : 16,
# #     }
# # cb.set_label('T' ,fontdict=font) #设置colorbar的标签字体及其大小
#
# # ax_cbar = fig.add_subplot(rect)
# # fig.colorbar(im,cax=ax_cbar,)

plt.savefig("pill.png", dpi=600,format="png")
plt.show()
