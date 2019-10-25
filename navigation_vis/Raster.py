import numpy as np
from matplotlib import cm as cm, pyplot as plt, colors as mplotcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from navigation_vis.AbstractView import AbstractView

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
         |   Y   1   |   2   |  ...  |   nX  |
         |   |       |       |       |       |
         |    ---X--- ------- ------- -------
         |   |       |       |       |       |
         |   |   2   |       |       |       |
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
        n_states = nY * nX
    elif n_dim == 4:
        # N x Y x X x C (array of color/gray images)
        image_grid = image_array_to_grid(data)
        nY, nX, Y, X, C = image_grid.shape
        flattened = image_grid_to_raster(image_grid)  # .transpose(1,0,2,3,4)
        n_states = len(data)
    elif n_dim == 5:
        image_grid = data
        nY, nX, Y, X, C = image_grid.shape
        flattened = image_grid_to_raster(image_grid)
        n_states = nY * nX
    else:
        raise Exception("data dimension {} not supported!".format(n_dim))

    return flattened, nY, nX, n_states

def get_discrete_cmap(cmap, discrete_levels):

    n_unique = len(np.unique(discrete_levels))
    norm = mplotcolors.Normalize(vmin=0, vmax=len(discrete_levels))
    # Leave string colors as it is, convert int colors to
    # normalized rgba
    cell_colors = [cmap(norm(val)) for val in discrete_levels]
    cmap = mplotcolors.ListedColormap(cell_colors, N=n_unique)
    return cmap

def get_css4_colors(N, shuffled=False):

    colors = list(mplotcolors.CSS4_COLORS.keys())

    if shuffled:
        np.random.shuffle(colors)

    if N == 0:
        return None
    elif N < 0:
        return colors

    times = int(np.ceil(N / len(colors)))

    if times == 1:
        return colors[:N]
    else:
        colors_tiled = colors * times
        return colors_tiled[:N]

class Raster(AbstractView):

    def __init__(self, data, ax=None):
        if ax is None:
            self.fig, self.ax = plt.subplots(1, 1)
        else:
            self.ax = ax
        self.update_data(data)

    def update_data(self, data):
        self.data = data
        self.raster, self.nY, self.nX, self.n_states = flatten_to_raster(data)
        self.H, self.W, self.C = self.raster.shape
        self.Y, self.X = self.H // self.nY, self.W // self.nX  # guaranteed to be ints
        self.render_img = self.raster[:, :, 0] if self.raster.shape[2] == 1 else self.raster
        # self.render_img = self.render_img[::-1]
        # self.render_img = np.roll(self.render_img, -1, axis=0)

    def _cell_coord_to_cell_idx(self, xi, yi):
        return yi * self.nX + xi

    def _xi_to_x(self, xi, center=True):
        x = xi * self.X + (self.X // 2 if center else 0)
        assert x >= 0 and x < self.W
        return x

    def _yi_to_y(self, yi, center=True):
        y = yi * self.Y + (self.Y // 2 if center else 0)
        assert y >= 0 and y < self.H
        return y

    def add_pixel_trajectory(self, trajectory, with_arrow=True, arrow_props=dict(), color='white'):
        x_list, y_list, a_list = [], [], []
        for (x, y, a) in trajectory:
            x_list.append(x)
            y_list.append(y)
        if with_arrow:
            self.draw_path_with_arrows(x_list, y_list, color, arrow_props)
        else:
            self.draw_path(x_list, y_list, color)
        return self

    def add_trajectory(self, trajectory, with_arrow=True, arrow_props=dict(), color='white'):
        x_list, y_list, a_list = [], [], []
        for (x, y, a) in trajectory:
            x_list.append(x)
            y_list.append(y)
        x_list = [self._xi_to_x(xi) for xi in x_list]
        y_list = [self._yi_to_y(yi) for yi in y_list]
        if with_arrow:
            self.draw_path_with_arrows(x_list, y_list, color, arrow_props)
        else:
            self.draw_path(x_list, y_list, color)
        return self

    def add_pixel_trajectories(self, trajectories, with_arrow=True, arrow_props=dict(), color='white'):
        traj_color_list = get_css4_colors(len(trajectories), True)
        for i, traj in enumerate(trajectories):
            self.add_pixel_trajectory(traj, with_arrow, arrow_props, traj_color_list[i])
        return self

    def add_trajectories(self, trajectories, with_arrow=True, arrow_props=dict(), color='white'):
        traj_color_list = get_css4_colors(len(trajectories), True)
        for i, traj in enumerate(trajectories):
            self.add_trajectory(traj, with_arrow, arrow_props, traj_color_list[i])
        return self

    def render(self, cmap=cm.viridis):
        self.cmap = cmap
        self.im = self.ax.imshow(self.render_img, cmap=cmap)
        # self.ax.invert_yaxis()
        return self

    def get_raster_entry(self, x, y):
        return self.render_img[y, x]

    def draw_path(self, x_list, y_list, color):
        return plt.plot(x_list, y_list, color=color)

    def draw_path_with_arrows(self, x_list, y_list, color, arrow_props):
        if len(x_list) >= 2:
            for i in range(len(x_list)-1):
                x, y = x_list[i], y_list[i]
                xp, yp = x_list[i+1], y_list[i+1]
                self.ax.annotate("", xy=(xp, yp), xytext=(x, y), arrowprops={"color":color, "arrowstyle": "->", **arrow_props})
        return self

    def ticks(self, xticks=True, yticks=True, minor=True):
        show_minor_ticks = False
        if self.nX == 1 and self.nY == 1:
            # one image
            xtick_locs = np.arange(0, self.W, 1) - 0.5
            ytick_locs = np.arange(0, self.H, 1) - 0.5
        else:
            # image grid
            # xtick_locs = np.arange(0, self.W, self.X) + self.X // 2 - (0.5 if self.X % 2 == 0 else 0)
            # ytick_locs = np.arange(0, self.H, self.Y) + self.Y // 2 - (0.5 if self.Y % 2 == 0 else 0)
            xtick_locs = np.arange(0, self.W, self.X) - 0.5
            ytick_locs = np.arange(0, self.H, self.Y) - 0.5
            xminor_tick_locs = np.arange(0, self.W, 1)
            yminor_tick_locs = np.arange(0, self.H, 1)
            show_minor_ticks = True

        if xticks:
            # set major ticks
            self.ax.set_xticks(xtick_locs, minor=False)
            self.ax.set_xticklabels([str(tick_idx) for tick_idx, tick_val in enumerate(xtick_locs)], minor=False)
            # set minor ticks
            if minor and show_minor_ticks:
                self.ax.set_xticks(xminor_tick_locs, minor=True)

        if yticks:
            # set major ticks
            self.ax.set_yticks(ytick_locs, minor=False)
            self.ax.set_yticklabels([str(tick_idx) for tick_idx, tick_val in enumerate(ytick_locs)], minor=False)
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

    def colorbar(self, ticks, ticklabels=None, fig=None):
        if fig is None:
            fig = self.fig
        if ticklabels is None:
            ticklabels = ticks
        divider = make_axes_locatable(self.ax)
        self.cb_ax = divider.append_axes('right', size='5%', pad=0.05)
        self.cbar = fig.colorbar(self.im, ticks=ticks, cax=self.cb_ax)
        self.cbar.set_ticklabels(ticklabels)
        return self

    def show_raster_text(self, fmt=".1f", color_cb=lambda elem: "white", fontsize=None):
        # Loop over data dimensions and add text annotations.
        for y in range(self.H):
            for x in range(self.W):
                if self.C == 1:
                    text =  format(self.get_raster_entry(x, y), fmt)
                else:
                    text = ", ".join([format(n, fmt) for n in self.get_raster_entry(x, y)])
                color = color_cb(self.get_raster_entry(x, y))
                # ax.text doesn't understand y axis inversion so do it for it
                self.ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fontsize)
        return self

    def show_cell_text(self, text_lst=None, fmt=".1f", color_cb=lambda elem: "white", fontsize=None):
        # Loop over cells and add text annotations.
        for yi in range(self.nY):
            for xi in range(self.nX):
                x, y = self._xi_to_x(xi), self._yi_to_y(yi)
                state_idx = self._cell_coord_to_cell_idx(xi, yi)
                if state_idx >= self.n_states:
                    continue
                if text_lst is None:
                    if self.C == 1:
                        text =  format(self.get_raster_entry(x, y), fmt)
                    else:
                        text = ", ".join([format(n, fmt) for n in self.get_raster_entry(x, y)])
                    color = color_cb(self.get_raster_entry(x, y))
                else:
                    text = text_lst[state_idx]
                    color = color_cb(state_idx)
                # ax.text doesn't understand y axis inversion so do it for it
                self.ax.text(x, y, text, ha="center", va="center", color=color, fontsize=fontsize)
        return self

    def __call__(self, *args, **kwargs):
        return self.raster