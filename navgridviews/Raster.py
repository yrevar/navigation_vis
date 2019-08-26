import numpy as np
from matplotlib import cm as cm, pyplot as plt
from navgridviews.AbstractView import AbstractView

def image_grid_to_raster(image_grid):
    """ Converts image grid to RGB raster image.
    :param image_grid: numpy array of size (nY x nX x Y x X x C)
    :return: RGB image
    """
    """
    1. nY    x    nX    x    Y    x    X    x    C        -> swap axes
                   .       .
                     .   .
                       .
                     .   .
                   .       .
    2. nY    x    Y    x    nX    x    X    x    C        -> reshape merge axis 0 with 1, 2 with 3
    3. (nY   *   Y)    x    (nX   *   X)    x    C        = H x W x C
    """
    nY, nX, Y, X, C = image_grid.shape
    return image_grid.swapaxes(  # 1
        1, 2).reshape(  # 2
        nY * Y, nX * X, C)

def image_array_to_grid(image_array, empty_fill=0):
    """ Converts image array to grid array.

    :param image_array: numpy array of size (N x Y x X x C)
    :param empty_fill: what to fill empty blocks with
    :return: numpy array of size (nY x nX x Y x X x C)
    """
    # Ref: https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
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
    """ Flatten numpy array of various dimensions to RGB raster image.
    :param data: numpy array of one of following sizes.
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
        self.Y, self.X = self.H // self.nY, self.W // self.nX # guaranteed to be ints
        self.ax = None

    def update_data(self, data):
        self.data = data

    def add_trajectory(self, trajectory):
        raise NotImplementedError

    def add_trajectories(self, trajectories):
        raise NotImplementedError

    def render(self, cmap=cm.viridis, ax=None):

        if ax is None:
            if self.ax is None:
                self.fig, self.ax = plt.subplots(1, 1)
        else:
            self.ax = ax
        render_img = self.raster[:,:,0] if self.raster.shape[2] == 1 else self.raster
        self.im = self.ax.imshow(render_img, cmap=cmap)
        return self

    def ticks(self, xticks=True, yticks=True, minor=True, coord_sys="rowcol"):

        show_minor_ticks = False
        # one image
        if self.nX == 1 and self.nY == 1:
            xtick_locs = np.arange(0, self.W, 1) - 0.5
            ytick_locs = np.arange(1, self.H+1, 1) - 0.5
        else:
            # xtick_locs = np.arange(0, self.W, self.X) + self.X // 2 - (0.5 if self.X % 2 == 0 else 0)
            # ytick_locs = np.arange(0, self.H, self.Y) + self.Y // 2 - (0.5 if self.Y % 2 == 0 else 0)
            xtick_locs = np.arange(0, self.W, self.X) - 0.5
            ytick_locs = np.arange(self.Y, self.H+self.Y, self.Y) - 0.5
            xminor_tick_locs = np.arange(0, self.W, 1)
            yminor_tick_locs = np.arange(0, self.H, 1)
            show_minor_ticks = True

        if coord_sys == "rowcol":
            x_origin, y_origin = 0, 0
        elif coord_sys == "cartesian":
            x_origin, y_origin = 1, 1
            ytick_locs = ytick_locs[::-1]
        else:
            raise Exception("coord system {} not supported!".format(coord_sys))

        if xticks:
            # set major ticks
            self.ax.set_xticks(xtick_locs, minor=False)
            self.ax.set_xticklabels([str(tick_idx + x_origin) for tick_idx, tick_val in enumerate(xtick_locs)], minor=False)
            # set minor ticks
            if minor and show_minor_ticks:
                self.ax.set_xticks(xminor_tick_locs, minor=True)

        if yticks:
            # set major ticks
            self.ax.set_yticks(ytick_locs, minor=False)
            self.ax.set_yticklabels([str(tick_idx + y_origin) for tick_idx, tick_val in enumerate(ytick_locs)], minor=False)
            # set minor ticks
            if minor and show_minor_ticks:
                self.ax.set_yticks(yminor_tick_locs, minor=True)

        # tick properties
        self.ax.tick_params(which='both', width=1)
        self.ax.tick_params(which='major', length=7, color='red')
        self.ax.tick_params(which='minor', length=2, color='black')
        return self

    def grid(self, major=True, minor=False):
        if major:
            self.ax.grid(which='major', linestyle='-', linewidth='1', color='red')
        if minor:
            self.ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')
        return self

    def title(self, title):
        self.ax.set_title(title)
        return self

    def __call__(self, *args, **kwargs):
        return self.raster
