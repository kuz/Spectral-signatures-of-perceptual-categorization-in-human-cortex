from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


# percentage
data_mono = np.array([[0.05263158, 0.0625    , 0.61111111, 1.        ],
                      [1.        , 0.66129032, 1.        , 1.        ],
                      [1.        , 0.76595745, 1.        , 1.        ],
                      [1.        , 1.        , 1.        , 1.        ],
                      [0.84210526, 0.08015267, 0.7312253 , 1.        ]]) * 100

data_poly = np.array([[0.94736842, 0.9375    , 0.38888889, 0.        ],
                      [0.        , 0.33870968, 0.        , 0.        ],
                      [0.        , 0.23404255, 0.        , 0.        ],
                      [0.        , 0.        , 0.        , 0.        ],
                      [0.15789474, 0.91984733, 0.2687747 , 0.        ]]) * 100

# absolutes
data_mono = np.array([[  1.,   4.,  33.,  82.],
                      [  8.,  41.,  99., 210.],
                      [ 30.,  72., 123., 110.],
                      [  6.,  28.,  51.,  55.],
                      [ 16.,  21., 370.,  31.]])

data_poly = np.array([[ 18.,  60.,  21.,   0.],
                      [  0.,  21.,   0.,   0.],
                      [  0.,  22.,   0.,   0.],
                      [  0.,   0.,   0.,   0.],
                      [  3., 241., 136.,   0.]])

column_names = ['0 ms','100 ms','250 ms','500 ms','1000 ms']
row_names = ['4Hz', '8 Hz','14 Hz','31 Hz','60 Hz', '150 Hz']

fig = plt.figure()
ax = Axes3D(fig)

lx = len(data_mono[0])
ly = len(data_mono[:,0])
xpos = np.arange(0,lx,1)
ypos = np.arange(0,ly,1)
xpos, ypos = np.meshgrid(xpos+0.3, ypos+0.1)

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(lx*ly)

dx = 0.25 * np.ones_like(zpos)
dy = 0.02 * np.ones_like(zpos)
dz_mono = data_mono.flatten()
dz_poly = data_poly.flatten()

ax.bar3d(xpos,ypos,zpos, dx, dy, dz_mono, color='r', alpha=0.5)
ax.bar3d(xpos+0.25,ypos,zpos, dx, dy, dz_poly, color='b', alpha=0.5)

#sh()
ax.w_xaxis.set_ticklabels(column_names)
ax.set_yticks([0.,  1., 2., 3., 4., 5.])
ax.set_xticks([0., 1., 2., 3., 4.])
ax.w_yaxis.set_ticklabels(row_names)
ax.set_zlabel('Number of features')

plt.show()