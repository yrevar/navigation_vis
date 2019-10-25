import numpy as np
from collections import namedtuple
from matplotlib import cm as cm, pyplot as plt, gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from navigation_vis.AbstractView import AbstractView


class GridSpec(AbstractView):
    MAX_PHI_DIM = 3
    CellParams = namedtuple("CellParams", "vmin vmax xlim ylim xticks yticks spines", 
                            defaults=[None, None, None, None, [], [], False])
    GridParams = namedtuple("GridParams", "xticks yticks spines", 
                            defaults=[None, None, True])
    def __init__(self, grid_data):
        """
        :params grid_values: array of size (H x W x D), where D is number of features

        """
        self.grid_data = None
        self.H, self.W = None, None
        self.X, self.Y = None, None
        self.feature_shape = None
        self.fig = None
        self.gs = None
        self.axes_grid = None
        self.update_data(grid_data)
        
    def update_data(self, grid_data):
        self.grid_data = grid_data
        self.H, self.W = self.grid_data.shape[:2]
        self.X, self.Y = self.W + 1, self.H + 1
        self.feature_shape = self.grid_data.shape[2:]
        self.fig = plt.figure(figsize=(min(self.W * 2, 20), min(self.H * 2, 20)))
        self.gs = gridspec.GridSpec(self.H, self.W)
        self.gs.update(wspace=0., hspace=0., left=0., right=1., bottom=0., top=1.)
        self.axes_grid = {
            self._rowcol_to_xy(row, col): plt.Subplot(self.fig, self.gs[row, col]) \
            for row in range(self.H) \
            for col in range(self.W) \
            }
        
    def get_axis_xy(self, x, y):
        return self.axes_grid[(x, y)]

    def set_axis_xy(self, x, y, ax):
        self.axes_grid[(x, y)] = ax

    def get_axis_rc(self, r, c):
        return self.axes_grid[self._rowcol_to_xy(r, c)]

    def set_axis_rc(self, r, c, ax):
        self.axes_grid[self._rowcol_to_xy(r, c)] = ax

    def get_cell_center(self):

        n_dim = len(self.feature_shape)
        if n_dim == 0 or (n_dim == 1 and self.feature_shape[0] == 1):
            return 0, 0
        elif n_dim == 1 and self.feature_shape[0] > 1:
            return self.feature_shape[0] / 2 - 0.5, 0.5
        elif 2 <= n_dim <= 3:
            return (self.feature_shape[0] / 2 - (1 - self.feature_shape[0] % 2) * 0.5,  # -0.5 when shape is even
                    self.feature_shape[1] / 2 - (1 - self.feature_shape[1] % 2) * 0.5)
        else:
            return 0, 0

    def _prepare_cell_axis(self, row, col, cell_params):
        """Prepares gridspec axes.

        """
        ax = self.get_axis_rc(row, col)

        if cell_params is None:
            cell_params = self.CellParams()
        for sp in ax.spines.values():
            sp.set_visible(cell_params.spines)
        if cell_params.xlim is not None:
            ax.set_xlim(cell_params.xlim)
        if cell_params.ylim is not None:
            ax.set_ylim(cell_params.ylim)
        if cell_params.xticks is not None:
            ax.set_xticks(cell_params.xticks)
        if cell_params.yticks is not None:
            ax.set_yticks(cell_params.yticks)
        self.set_axis_rc(row, col, ax)
        return ax, cell_params

    def _prepare_grid_axis(self, row, col, grid_params):
        """Prepares gridspec axes.

        """
        ax = self.get_axis_rc(row, col)
        if grid_params is None:
            grid_params = self.GridParams()

        if ax.is_first_row():
            ax.spines['top'].set_visible(grid_params.spines)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(grid_params.spines)
            if grid_params.xticks is not None:
                ax.set_xticks(grid_params.xticks)
        if ax.is_first_col():
            ax.spines['left'].set_visible(grid_params.spines)
            if grid_params.yticks is not None:
                ax.set_yticks(grid_params.yticks)
        if ax.is_last_col():
            ax.spines['right'].set_visible(grid_params.spines)
        self.set_axis_rc(row, col, ax)
        return ax, grid_params

    def _connect_axes(self, ax1, ax2):
        """Draws an arrow from ax1 to ax2.

        """
        axis_center = self.get_cell_center()
        con = ConnectionPatch(xyA=axis_center, xyB=axis_center,
                              coordsA="data", coordsB="data",
                              axesA=ax1, axesB=ax2, color="red",
                              mutation_scale=40, arrowstyle="->",
                              shrinkB=5, shrinkA=5)
        ax1.add_artist(con)
        con = ConnectionPatch(xyA=axis_center, xyB=axis_center,
                              coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color="red",
                              mutation_scale=40, arrowstyle="<-",
                              shrinkB=5, shrinkA=5)
        ax2.add_artist(con)
        ax1.plot(*axis_center, 'ro', markersize=10)
        ax2.plot(*axis_center, 'ro', markersize=10)

    def _xy_to_rowcol(self, x, y):
        """Converts (x, y) to (row, col).

        """
        return self.H - y, x - 1

    def _rowcol_to_xy(self, row, col):
        """Converts (row, col) to (x, y).

        """
        return col + 1, self.H - row

    def render_plot(self, row, col, cell_params, grid_params):
        """Render a cell of of the grid as subplot.

        """
        feature = self.grid_data[row, col]
        ax, cell_params = self._prepare_cell_axis(row, col, cell_params)
        ax, grid_params = self._prepare_grid_axis(row, col, grid_params)

        n_dim = len(feature.shape)
        if n_dim == 0 or (n_dim == 1 and feature.shape[0] == 1):
            ax.imshow(np.atleast_2d(feature), vmin=cell_params.vmin, vmax=cell_params.vmax)
        elif n_dim == 1 and feature.shape[0] > 1:
            ax.plot(feature)
        elif 1 < n_dim <= 3:
            ax.imshow(feature, vmin=cell_params.vmin, vmax=cell_params.vmax, cmap="gray" if n_dim == 2 else None)
        else:
            raise ValueError("Feature dimensions > {} not supported!".format(self.MAX_PHI_DIM))

        return ax

    def prepare_plots(self, cell_params=None, grid_params=None):
        for row in range(self.H):
            for col in range(self.W):
                self.render_plot(row, col, cell_params, grid_params)

    def render(self, cell_params=None, grid_params=None):
        """Render the grid
        """
        self.prepare_plots(cell_params, grid_params)
        for row in range(self.H):
            for col in range(self.W):
                self.fig.add_subplot(self.get_axis_rc(row, col))
        return self

    def add_trajectory(self, trajectory):
        """Adds a trajectory to the figure.

        :params trajectory: [(x1, y1), ...]
        """
        x_list, y_list = tuple(zip(*trajectory))  # [(x, y), ..] -> [x, ...], [y, ...]
        for idx in range(len(x_list) - 1):
            ax1 = self.axes_grid[(x_list[idx], y_list[idx])]
            ax2 = self.axes_grid[(x_list[idx + 1], y_list[idx + 1])]
            ax1.set_zorder(-2 * idx + 1)
            ax2.set_zorder(-2 * idx)
            self._connect_axes(ax1, ax2)

    def add_trajectories(self, traj_lst):
        """Adds a list of trajectories to the figure.

        :params traj_lst: list of trajectories, where trajectory = [(x1, y1), ...]
        """
        if traj_lst is not None:
            for traj in traj_lst:
                self.add_trajectory(traj)