import numpy as np
from matplotlib import cm as cm, pyplot as plt
from navgridviews.AbstractView import AbstractView

def image_grid_to_raster(image_grid):
    """

    1. nY    x    nX    x    Y    x    X    x    C        -> swap axes
                   .       .
                     .   .
                       .
                     .   .
                   .       .
    2. nY    x    Y    x    nX    x    X    x    C        -> reshape merge axis 0 with 1, 2 with 3
    3. (nY   *   Y)    x    (nX   *   X)    x    C        = H x W x C
    :return:
    """
    nY, nX, Y, X, C = image_grid.shape
    return image_grid.swapaxes(  # 1
        1, 2).reshape(  # 2
        nY * Y, nX * X, C)

def image_array_to_grid(image_array, empty_fill=0):

    N, Y, X, C = image_array.shape
    #  N    x    Y     x    X    x    C        -> factor axis 1 into two axis
    # nY    x    nX    x    Y    x    X    x    C
    nX = int(np.ceil(np.sqrt(N)))
    nY = int(np.ceil(N / nX))  # nX >= nY
    n_empty_blocks = (nX * nY) - N
    if n_empty_blocks > 0:
        image_array = np.concatenate(
            [image_array,
             empty_fill * np.ones((n_empty_blocks, Y, X, C), dtype=image_array.dtype)
             ]
        )
    return image_array.reshape(nY, nX, Y, X, C)

def flatten_to_raster(data):
    """
       :param data: Data format can be
                   1) H x W x C (color/gray image)
                   2) N x Y x X x C (array of color/gray images)
                   3) nY x nX x Y x X x C (2d array of color/gray images)
                   (C has to be 1 or 3)
                   E.g., C = 1
                     <---------------W--------------->
                      ------- ------- ------- -------
                 ^   |       |       |       |       |
                 |   Y   0   |   1   |  ...  |   nX  |
                 |   |       |       |       |       |
                 |    ---X--- ------- ------- -------
                 |   |       |       |       |       |
                 |   |   1   |       |       |       |
                 |   |       |       |       |       |
                 H    ------- ------- ------- -------
                 |   |       |       |       |       |
                 |   |  ...  |       |       |       |
                 |   |       |       |       |       |
                 |    ------- ------- ------- -------
                 |   |       |       |       |       |
                 |   |  nY   |       |       |       |
                 v   |       |       |       |       |
                      ------- ------- ------- -------
               Input -> Output
               1) H x W x C -> H x W x C
               2) N x Y x X x C -> (nY*Y) x (nX*X) x C
                   where nX & nY are factors of N such that we get as close to square grid as possible
                   (with bias towards having more columns than rows so for 12 images we have nY x nX = 3 x 4 grid)
               3) nH x nW x H x W x C -> (nY*Y) x (nX*X) x C
    """
    n_dim = len(data.shape)
    if n_dim == 3:
        H, W, C = data.shape
        nY, nX = 1, 1
        flattened = data
    elif n_dim == 4:
        # N x Y x X x C (array of color/gray images)
        image_grid = image_array_to_grid(data)
        nY, nX, Y, X, C = image_grid.shape
        flattened = image_grid_to_raster(image_grid)  # .transpose(1,0,2,3,4)
    elif n_dim == 5:
        image_grid = data
        nY, nX, Y, X, C = image_grid.shape
        flattened = image_grid_to_raster(image_grid)
    else:
        raise Exception("data dimension {} not supported!".format(n_dim))

    return flattened, nY, nX

class Raster(AbstractView):
    def __init__(self, data):

        self.data = data
        self.raster, self.nY, self.nX = flatten_to_raster(data)
        self.H, self.W, self.C = self.raster.shape

    def update_data(self, data):
        self.data = data

    def add_trajectory(self, trajectory):
        raise NotImplementedError

    def add_trajectories(self, trajectories):
        raise NotImplementedError

    def render(self):
        plt.imshow(self.raster)

    def __call__(self, *args, **kwargs):
        return self.raster
