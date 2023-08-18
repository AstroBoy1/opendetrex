# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import numpy as np
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import inverse_sigmoid

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb

import cv2 as cv
from torchvision import transforms
from torchvision.transforms import GaussianBlur

from torch.autograd import Variable
#from fast_bilateral import *

# Math libraries
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spalg
import scipy.stats as stats
import itertools

# Plotting libraries
import matplotlib.pyplot as plt

# Image libraries
import matplotlib.image as mpimg
from PIL import Image
import scipy.ndimage as ndimage
import skimage.transform

# Progress bar
#from tqdm import tqdm, trange

def get_coord_from_vertex(v, grid_shape):
    """Get the coordinates of a grid vertex from its index."""
    v = np.array(v)
    grid_shape = np.array(grid_shape)
    prod = np.cumprod(grid_shape[::-1])[::-1][1:][:, None]
    first_coord = (v // prod) % grid_shape[:-1][:, None]
    last_coord = v % grid_shape[-1]
    return np.vstack([first_coord, last_coord]).T

def get_vertex_from_coord(coord, grid_shape):
    """Get the index of a grid vertex from its coordinates."""
    coord = np.array(coord)
    grid_shape = np.array(grid_shape)
    prod = np.cumprod(grid_shape[::-1])[::-1][1:][:, None]
    return (coord[:, :-1].dot(prod) + coord[:, -1:])[:, 0]


class BilateralGrid:

    def __init__(self, ref_img, sigma):
        """Initialize a bilateral grid with a reference image and a standard deviation vector."""
        #breakpoint()
        self.ref_img = ref_img
        self.sigma = sigma

        self.compute_everything()

    def compute_everything(self):
        """Perform all computations necessary for the fast bilateral solver."""
        #breakpoint()
        self.compute_useful_stuff()
        self.compute_splat()
        self.compute_blur()
        self.compute_bistochastization()
        self.compute_pyramid_space()

    def compute_useful_stuff(self):
        """Translate the pixels of a 2d image into D-dimensional positions."""
        # Spatial coordinates of all the image pixels
        #breakpoint()
        self.x_ref_img = np.indices(self.ref_img.shape[:2])[0].flatten()
        self.y_ref_img = np.indices(self.ref_img.shape[:2])[1].flatten()

        # Positions (coordinates + values) of all the image pixels
        self.pos_ref_img = np.hstack([
            self.x_ref_img[:, None],
            self.y_ref_img[:, None],
            self.ref_img[self.x_ref_img, self.y_ref_img]
        ])

        # Dimension of the position: 2 + number of channels
        self.D = 2 + self.ref_img.shape[2]

        # Shape of the D-dimensional bilateral grid:
        # - sizes of the coordinate axes
        # - sizes of the value axes
        self.grid_shape = np.hstack([
            np.ceil(self.ref_img.shape[:2] / self.sigma[:2]) + 1,
            np.ceil(self.ref_img.max() / self.sigma[2:]) + 1
        ]).astype(int)

        # Number of pixels and vertices
        self.n_pixels = np.prod(self.ref_img.shape[:2])
        self.n_vertices = np.prod(self.grid_shape)

    def compute_splat(self):
        """Compute the splat matrix: links pixels to the associated vertices in the grid."""
        #tqdm(desc="Splat matrix computation")

        # Positions of the nearest neighbor vertices associated with each image pixel
        self.pos_grid = np.rint(self.pos_ref_img / self.sigma).astype(int)
        # Indices of the nearest neighbor vertices
        self.nearest_neighbors = get_vertex_from_coord(self.pos_grid, self.grid_shape)

        # Vertices that are nearest neighbor to at least one pixel
        # (all the other vertices of the grid are useless)
        self.useful_vertices = np.sort(np.unique(self.nearest_neighbors))
        self.n_useful_vertices = len(self.useful_vertices)

        # Dictionary of coordinates for useful vertices
        self.useful_vertex_to_coord = get_coord_from_vertex(self.useful_vertices, self.grid_shape)
        # Dictionary of indices for useful vertices
        self.useful_vertex_to_ind = np.empty(self.n_vertices)
        self.useful_vertex_to_ind[self.useful_vertices] = np.arange(self.n_useful_vertices)

        # Record if a given vertex is useful (comes in handy for slicing)
        self.vertex_is_useful = np.zeros(self.n_vertices)
        self.vertex_is_useful[self.useful_vertices] = 1

        # Construction of the splat matrix: (vertex, pixel) => neighbor?
        row_ind = self.useful_vertex_to_ind[self.nearest_neighbors]
        col_ind = np.arange(self.n_pixels)
        data = np.ones_like(row_ind)
        self.S = sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(self.n_useful_vertices, self.n_pixels)
        )

    def compute_blur(self):
        """Compute the blur matrix: superposition of (1, 2, 1) filters on the bilateral grid."""
        B = sparse.lil_matrix((self.n_useful_vertices, self.n_useful_vertices))
        # Fill the diagonal of the blur matrix
        B[np.arange(self.n_useful_vertices), np.arange(self.n_useful_vertices)] = 2 * self.D

        # List all the +-1 coordinate changes possible in D dimensions
        possible_neighbor_steps = [
                                      np.array([0] * dim + [1] + [0] * (self.D - dim - 1))
                                      for dim in range(self.D)
                                  ] + [
                                      np.array([0] * dim + [-1] + [0] * (self.D - dim - 1))
                                      for dim in range(self.D)
                                  ]

        for neighbor_step in possible_neighbor_steps:
        #for neighbor_step in tqdm(possible_neighbor_steps, desc="Blur matrix computation"):
            # Compute the +-1 neighbors only for the useful vertices
            neighbors_coord = self.useful_vertex_to_coord + neighbor_step

            # Check whether these neighbors are still in the grid
            neighbors_exist = True
            for dim, dim_size in enumerate(self.grid_shape):
                neighbors_exist = (
                        neighbors_exist &
                        (neighbors_coord[:, dim] >= 0) &
                        (neighbors_coord[:, dim] < dim_size)
                )

            # Select only the vertices whose neighbors are still in the grid
            vertices_with_existing_neighbors = self.useful_vertices[neighbors_exist]
            existing_neighbors_coord = neighbors_coord[neighbors_exist]
            existing_neighbors_vertices = get_vertex_from_coord(existing_neighbors_coord, self.grid_shape)

            # Select only the vertices whose neighbors are also useful
            neighbors_among_useful = self.vertex_is_useful[existing_neighbors_vertices].astype(bool)
            vertices_with_useful_neighbors = vertices_with_existing_neighbors[neighbors_among_useful]
            useful_neighbors_vertices = existing_neighbors_vertices[neighbors_among_useful]

            # Construct splat matrix: (vertex, vertex) => filter coefficient
            row_ind = self.useful_vertex_to_ind[vertices_with_useful_neighbors]
            col_ind = self.useful_vertex_to_ind[useful_neighbors_vertices]
            B[row_ind, col_ind] = 1

        self.B = B.tocsr()

    def compute_bistochastization(self, iterations=20):
        """Compute diagonal bistochastization matrices."""
        #tqdm(desc="Bistochastization")
        m = self.S.dot(np.ones(self.S.shape[1]))
        n = np.ones(self.B.shape[1])
        for it in range(iterations):
            new_n = np.sqrt((n * m) / (self.B.dot(n)))
            if np.linalg.norm(new_n - n) < 1e-5:
                break
            else:
                n = new_n
        Dn = sparse.diags(n)
        Dm = sparse.diags(m)

        self.Dn, self.Dm = Dn, Dm

    def compute_pyramid_space(self):
        """Compute pyramidal decomposition."""
        self.S_pyr = build_pyramid(self.useful_vertex_to_coord)
        self.P = build_P(self.S_pyr)
        self.z_weight_init = build_z_weight(self.S_pyr, alpha=4, beta=0)
        self.z_weight_precond = build_z_weight(self.S_pyr, alpha=2, beta=5)

def prec_conj_grad(A, b, init, M_1, channel=0, iterations=25):
    """Perform preconditioned conjugate gradient descent."""
    x = init
    r = b - A.dot(x)
    d = M_1(r)
    delta_new = r.dot(d)
    for it in range(iterations):
    #for it in trange(iterations, desc="Conjugate gradient - channel {}".format(channel)):
        q = A.dot(d)
        alpha = delta_new / d.dot(q)
        x = x + alpha * d
        r = r - alpha * q
        s = M_1(r)
        delta_old = delta_new
        delta_new = r.dot(s)
        beta = delta_new / delta_old
        d = s + beta * d
    return x

def M_jacobi(y, A):
    print("jac")
    return y / A.diagonal()


def bilateral_representation(V, sigma):
    """Compute a bilateral splat matrix for a matrix V of abstract pixels positions."""
    D = V.shape[1]
    grid_shape = np.ceil((V.max(axis=0) / sigma) + 1).astype(int)

    n_abstract_pixels = len(V)
    n_vertices = np.prod(grid_shape)

    pos_grid = np.rint(V / sigma).astype(int)

    # Positions of the nearest neighbor vertices associated with each abstract pixel
    nearest_neighbors = get_vertex_from_coord(pos_grid, grid_shape)
    # Vertices that are nearest neighbor to at least one pixel
    # (all the other vertices of the grid are useless)
    useful_vertices = np.sort(np.unique(nearest_neighbors))
    n_useful_vertices = len(useful_vertices)

    # Dictionary of indices for useful vertices
    useful_vertex_to_ind = np.empty(n_vertices)
    useful_vertex_to_ind[useful_vertices] = np.arange(n_useful_vertices)

    # Construction of the splat matrix: (vertex, abstract pixel) => neighbor?
    row_ind = useful_vertex_to_ind[nearest_neighbors]
    col_ind = np.arange(n_abstract_pixels)
    data = np.ones_like(row_ind)
    S = sparse.csr_matrix(
        (data, (row_ind, col_ind)), shape=(n_useful_vertices, n_abstract_pixels)
    )

    # Positions of the useful vertices from the grid
    new_V = get_coord_from_vertex(useful_vertices, grid_shape)

    return S, new_V

def build_pyramid(useful_vertex_to_coord):
    """Construct a pyramid of ever coarser splat matrices."""
    #tqdm(desc="Pyramid space construction")
    V = useful_vertex_to_coord
    S_pyr = []
    while len(V) > 1:
        Sk, V = bilateral_representation(V, 2 * np.ones(V.shape[1]))
        S_pyr.append(Sk)
    return S_pyr

def build_P(S_pyr):
    """Deduce the hierarchical projection matrix from the pyramid of splat matrices."""
    prod = sparse.eye(S_pyr[0].shape[1])
    P = prod
    for s in S_pyr:
        prod = s.dot(prod)
        P = sparse.vstack([P, prod])
    return P

def build_z_weight(S_pyr, alpha, beta):
    """Compute weights for all the stages of the pyramid space."""
    z_weight = np.ones(S_pyr[0].shape[1])
    for k, s in enumerate(S_pyr):
        z_weight = np.hstack([
            z_weight,
            (alpha ** (- beta - k - 1)) * np.ones(s.shape[0])
        ])
    return z_weight


def M_hier(y, A, P, z_weight):
    """Compute hierarchical preconditioner."""
    z_size, y_size = P.shape

    P1 = P.dot(np.ones(y_size))
    Py = P.dot(y)
    PA = P.dot(A.diagonal())

    return P.T.dot(z_weight * P1 * Py / PA)


def y_hier(S, C, T, P, z_weight):
    """Compute hierarchical initialization."""
    z_size, y_size = P.shape

    P1 = P.dot(np.ones(y_size))
    PSc = P.dot(S.dot(C))
    PSct = P.dot(S.dot(C * T))

    y_init = (
            P.T.dot(z_weight * PSct / P1) /
            P.T.dot(z_weight * PSc / P1)
    )

    return y_init


def solve(bilateral_grid, C, T, lambd, precond_init_method="hierarchical", channel=0):
    """Solve a least squares problem
    from its bistochastized splat-blur-slice decomposition
    using the preconditioned conjugate gradient."""
    # Retrieve information from the bilateral grid object
    S, B = bilateral_grid.S, bilateral_grid.B
    Dn, Dm = bilateral_grid.Dn, bilateral_grid.Dm

    # Compute the coefficients of the least-squares problem min Ax^2 + bx + c
    A = lambd * (Dm - Dn.dot(B).dot(Dn)) + sparse.diags(S.dot(C))
    b = S.dot(C * T)
    c = 0.5 * (C * T).dot(T)

    # Apply chosen preconditioning and initialization
    if precond_init_method == "simple":
        # Define initial vector and preconditioning function
        y_init = S.dot(C * T) / np.clip(S.dot(C), a_min=1, a_max=None)

        def M_1(y):
            return M_jacobi(y, A)

    elif precond_init_method == "hierarchical":
        # Retrieve pyramid information from the bilateral grid object
        P = bilateral_grid.P
        z_weight_init = bilateral_grid.z_weight_init
        z_weight_precond = bilateral_grid.z_weight_precond
        # Define initial vector and preconditioning function
        y_init = y_hier(S, C, T, P, z_weight_init)

        def M_1(y):
            return M_hier(y, A, P, z_weight_precond)

    else:
        raise ValueError("Wrong preconditioning")

    # Compute the optimal solution
    y_opt = prec_conj_grad(A, b, init=y_init, M_1=M_1, channel=channel)
    return y_opt


# In each function, ct is the 1d domain transform used to measure the distance between the pixel positions

def box_filter_naive(I, ct, sigma_H):
    """Apply a box filter with a naive for loop."""
    r = sigma_H * np.sqrt(3)
    J = np.empty_like(I)

    dim = I.shape[0]
    for p in range(dim):
        J_p = 0
        K_p = 0
        for q in range(p, dim):
            if abs(ct[q] - ct[p]) > r:
                break
            J_p += I[q, :]
            K_p += 1
        for q in range(p - 1, -1):
            if abs(ct[q] - ct[p]) > r:
                break
            J_p += I[q, :]
            K_p += 1
        J[p, :] = J_p[:] / K_p
    return J


def box_filter_recursive1(I, ct, sigma_H):
    """Apply a recursive box filter."""
    a = np.exp(-np.sqrt(2) / sigma_H)
    dim, channels = I.shape

    J_tmp = np.empty_like(I)
    J_tmp[0, :] = I[0, :]
    for p in range(1, dim):
        d = ct[p] - ct[p - 1]
        J_tmp[p, :] = (1 - a ** d) * I[p, :] + (a ** d) * J_tmp[p - 1, :]

    J = np.empty_like(I)
    J[-1, :] = J_tmp[-1, :]
    for p in range(dim - 2, -1, -1):
        d = ct[p + 1] - ct[p]
        J[p, :] = (1 - a ** d) * J_tmp[p, :] + (a ** d) * J[p + 1, :]

    return J


def box_filter_recursive_sparse(I, ct, sigma_H):
    """Apply a recursive box filter with sparse matrices for faster computation."""
    a = np.exp(-np.sqrt(2) / sigma_H)
    d = np.diff(ct)
    J = np.empty_like(I)

    dim, channels = I.shape

    A_forward = sparse.diags([1] + list(1 - a ** d))
    B_forward = sparse.identity(dim) - sparse.diags(a ** d, -1)

    A_backward = sparse.diags(list(1 - a ** d) + [1])
    B_backward = sparse.identity(dim) - sparse.diags(a ** d, 1)

    for channel in range(channels):
        J[:, channel] = spalg.spsolve(B_forward, A_forward.dot(I[:, channel]))
        J[:, channel] = spalg.spsolve(B_backward, A_backward.dot(J[:, channel]))

    return J

def smooth_cols(I, sigma_s, sigma_r, it, I_ref=None, N_it=3):
    """Apply a bilateral filter on all columns of an image."""
    if I_ref is None:
        I_ref = I
    # Compute the current spatial std of the filter
    sigma_H = sigma_s * np.sqrt(3) * (2 ** (N_it - it)) / np.sqrt(4 ** N_it - 1)
    # Intialize the new image
    new_I = np.empty_like(I)
    # Compute the vertical spatial derivative of the image channels
    I_ref_prime = np.vstack([I_ref[:1, :, :], np.diff(I_ref, axis=0)])
    for col in range(I.shape[1]):
    #for col in trange(I.shape[1], desc="Domain transform - iteration {} - columns".format(it)):
        # Compute the domain transform
        ct = np.cumsum(1 + (sigma_s / sigma_r) * np.abs(I_ref_prime[:, col, :].sum(axis=1)))
        # Apply the bilateral box filter
        new_I_slice = box_filter_recursive_sparse(I[:, col, :], ct, sigma_H)
        # Fill the column of the new image
        new_I[:, col, :] = new_I_slice
    return new_I

def smooth_rows(I, sigma_s, sigma_r, it, I_ref=None, N_it=3):
    """Apply a bilateral filter on all rows of an image."""
    if I_ref is None:
        I_ref = I
    sigma_H = sigma_s * np.sqrt(3) * (2 ** (N_it - it)) / np.sqrt(4 ** N_it - 1)
    # Intialize the new image
    new_I = np.empty_like(I)
    # Compute the horizontal spatial derivative of the image channels
    I_ref_prime = np.hstack([I_ref[:, :1, :], np.diff(I_ref, axis=1)])
    for row in range(I.shape[0]):
    #for row in trange(I.shape[0], desc="Domain transform - iteration {} - rows".format(it)):
        # Compute the domain transform
        ct = np.cumsum(1 + (sigma_s / sigma_r) * np.abs(I_ref_prime[row, :, :].sum(axis=1)))
        # Apply the bilateral box filter
        new_I_slice = box_filter_recursive_sparse(I[row, :, :], ct, sigma_H)
        # Fill the row of the new image
        new_I[row, :, :] = new_I_slice
    return new_I

def domain_transform(I0, sigma_s, sigma_r, I_ref=None, N_it=3):
    """Apply the domain transform to I0 with spatial std sigma_s and value std sigma_r."""
    I = I0.copy()
    for it in range(1, N_it+1):
        for axis in [0, 1]:
            if axis == 1:
                I = smooth_rows(I, sigma_s, sigma_r, it, I_ref=I_ref, N_it=N_it)
            else:
                I = smooth_cols(I, sigma_s, sigma_r, it, I_ref=I_ref, N_it=N_it)
    return I


def smoothing(
        ref_img,
        # Bilateral solver parameters
        lambd, sigma_xy, sigma_l=None, sigma_rgb=None,
        # Domain transform parameters
        sigma_s=None, sigma_r=None, dt_it=None,
        # Visualization parameters
        show=True, file_name=None,
        # Add sharpening result
        sharp=False
):
    #breakpoint()
    # Choose the right set of standard deviations
    sigma = None
    if ref_img.shape[2] == 1:
        sigma = np.array([sigma_xy, sigma_xy, sigma_l])
    elif ref_img.shape[2] == 3:
        sigma = np.array([sigma_xy, sigma_xy, sigma_rgb, sigma_rgb, sigma_rgb])

    # Create the bilateral grid
    bilateral_grid = BilateralGrid(ref_img, sigma)
    S, B = bilateral_grid.S, bilateral_grid.B

    # The target image is the same as the reference image
    target_img = ref_img
    new_img = np.empty_like(target_img)

    # Perform smoothing channel by channel
    for channel in range(target_img.shape[2]):
        T = target_img[:, :, channel].flatten()
        # The confidence in each pixel of the target is the same
        C = np.ones_like(T)

        # Compute bilateral space solution
        y = solve(bilateral_grid, C, T, lambd, channel=channel)
        # Go back to pixel space
        x = S.T.dot(y).reshape(ref_img.shape[:2])

        # Fill the corresponding channel of the new image
        new_img[:, :, channel] = x

    # Apply domain transform if needed
    if sigma_s is not None and sigma_r is not None:
        new_img = domain_transform(
            I0=new_img, I_ref=new_img,
            sigma_s=sigma_s, sigma_r=sigma_r, N_it=dt_it,
        )

    # Visualize and/or save
    if show:
        show_smoothing(ref_img, new_img, sharp=sharp)
    if file_name is not None:
        save_smoothing(ref_img, new_img, file_name, sharp=sharp)

    return new_img


def colorization(
        ref_img, target_img,
        lambd, sigma_xy, sigma_l,
        sigma_s=None, sigma_r=None, dt_it=None,
        show=True, file_name=None
):
    sigma = np.array([sigma_xy, sigma_xy, sigma_l])

    # Create the bilateral grid
    bilateral_grid = BilateralGrid(ref_img, sigma)
    S, B = bilateral_grid.S, bilateral_grid.B

    # The new image will have LUV channels
    new_img = np.empty_like(target_img)
    # The luminance channel is the same as in the reference black and white image
    new_img[:, :, 0] = ref_img[:, :, 0]

    for channel in range(1, 3):
        T = target_img[:, :, channel].flatten()
        # The confidence in the target pixels is zero...
        C = np.zeros_like(T)
        # ... except for those that have a color mark
        C[T != np.median(T)] = 1

        # Compute bilateral space solution
        y = solve(bilateral_grid, C, T, lambd, channel=channel)
        # Go back to pixel space
        x = S.T.dot(y).reshape(ref_img.shape[:2])

        # Fill the corresponding channel of the new image
        new_img[:, :, channel] = x

    # Apply domain transform if needed
    if sigma_s is not None and sigma_r is not None:
        new_img = domain_transform(
            I0=new_img, I_ref=ref_img,
            sigma_s=sigma_s, sigma_r=sigma_r, N_it=dt_it,
        )

    # Visualize and/or save
    if show:
        show_colorization(ref_img, target_img, new_img)
    if file_name is not None:
        save_colorization(ref_img, target_img, new_img, file_name)

    return new_img


def depth_superresolution(
        ref_img, target_img, f,
        lambd, sigma_xy, sigma_l, sigma_uv,
        sigma_s=None, sigma_r=None, dt_it=None,
        show=True, file_name=None
):
    sigma = np.array([sigma_xy, sigma_xy, sigma_l, sigma_uv, sigma_uv])
    # Create the bilateral grid
    bilateral_grid = BilateralGrid(ref_img, sigma)
    S, B = bilateral_grid.S, bilateral_grid.B

    # The confidence in each target pixel is a gaussian bump dictated by the upsampling factor
    T = target_img.flatten()
    C_block = stats.norm.pdf(
        x=np.indices((f, f))[0] + np.indices((f, f))[1],
        loc=0, scale=f / 4
    )
    C = np.tile(C_block, (target_img.shape[0] // f, target_img.shape[1] // f)).flatten()

    # Compute bilateral space solution
    y = solve(bilateral_grid, C, T, lambd, channel=0)
    # Go back to pixel space
    x = S.T.dot(y).reshape(ref_img.shape[:2])

    # Fill luminance channel of the new image
    new_img = x[:, :, None]

    # Apply domain transform if needed
    if sigma_s is not None and sigma_r is not None:
        new_img = domain_transform(
            I0=new_img, I_ref=ref_img,
            sigma_s=sigma_s, sigma_r=sigma_r, N_it=dt_it
        )

    # Visualize and/or save
    if show:
        show_depth_superresolution(ref_img, target_img, new_img)
    if file_name is not None:
        save_depth_superresolution(ref_img, target_img, new_img, file_name)

    return new_img

def normalize(img):
    m = img.min()
    M = img.max()
    return (255 * (img - m) / (M - m)).astype(int)


def show_smoothing(ref_img, new_img, sharp=False):
    new_img = np.clip(new_img, a_min=0, a_max=255)
    if ref_img.shape[2] == 1:
        ref_img, new_img = ref_img[:, :, 0], new_img[:, :, 0]
        cmap = "Greys_r"
    else:
        cmap = None
    if not sharp:
        fig, ax = plt.subplots(1, 2, figsize=(18, 9))
        ax[0].imshow(ref_img, cmap=cmap, vmin=0, vmax=255)
        ax[0].set_title("Original image")
        ax[1].imshow(new_img, cmap=cmap, vmin=0, vmax=255)
        ax[1].set_title("Smoothed image")
        plt.show()
    elif sharp:
        sharp_img = np.clip(ref_img + (ref_img - new_img), a_min=0, a_max=255)
        fig, ax = plt.subplots(1, 3, figsize=(18, 9))
        ax[0].imshow(ref_img, cmap=cmap, vmin=0, vmax=255)
        ax[0].set_title("Original image")
        ax[1].imshow(new_img, cmap=cmap, vmin=0, vmax=255)
        ax[1].set_title("Smoothed image")
        ax[2].imshow(sharp_img, cmap=cmap, vmin=0, vmax=255)
        ax[2].set_title("Sharpened image")
        plt.show()


def save_smoothing(ref_img, new_img, file_name, sharp=False):
    new_img = np.clip(new_img, a_min=0, a_max=255)
    if ref_img.shape[2] == 1:
        ref_img, new_img = ref_img[:, :, 0], new_img[:, :, 0]
    ref_img_pil = Image.fromarray(ref_img.astype("uint8"))
    ref_img_pil.save("results/smoothing_" + file_name + "_ref.png")
    new_img_pil = Image.fromarray(new_img.astype("uint8"))
    new_img_pil.save("results/smoothing_" + file_name + "_new.png")
    if sharp:
        sharp_img = np.clip(ref_img + (ref_img - new_img), a_min=0, a_max=255)
        sharp_img_pil = Image.fromarray(sharp_img.astype("uint8"))
        sharp_img_pil.save("results/smoothing_" + file_name + "_sharp.png")


def show_colorization(ref_img, target_img, new_img):
    new_img = np.clip(new_img, a_min=0, a_max=255)
    fig, ax = plt.subplots(1, 3, figsize=(18, 9))
    ax[0].imshow(Image.fromarray(ref_img[:, :, 0], mode="L").convert("RGB"))
    ax[0].set_title("Original BW image")
    ax[1].imshow(Image.fromarray(target_img, mode="YCbCr").convert("RGB"))
    ax[1].set_title("User-marked BW image")
    ax[2].imshow(Image.fromarray(new_img, mode="YCbCr").convert("RGB"))
    ax[2].set_title("Automatically colorized image")
    plt.show()


def save_colorization(ref_img, target_img, new_img, file_name):
    ref_img_pil = Image.fromarray(ref_img[:, :, 0].astype("uint8"), mode="L")
    ref_img_pil.save("results/colorization_" + file_name + "_ref.png")
    target_img_pil = Image.fromarray(target_img.astype("uint8"), mode="YCbCr").convert("RGB")
    target_img_pil.save("results/colorization_" + file_name + "_target.png")
    new_img_pil = Image.fromarray(new_img.astype("uint8"), mode="YCbCr").convert("RGB")
    new_img_pil.save("results/colorization_" + file_name + "_new.png")


def show_depth_superresolution(ref_img, target_img, new_img):
    new_img = np.clip(new_img, a_min=0, a_max=255)
    fig, ax = plt.subplots(1, 3, figsize=(18, 9))
    ax[0].imshow(np.asarray(Image.fromarray(ref_img.astype("uint8"), mode="YCbCr").convert("RGB")))
    ax[0].set_title("Reference color scene")
    ax[1].imshow(target_img[:, :, 0], cmap="Greys_r")
    ax[1].set_title("Noisy low-resolution depth map")
    ax[2].imshow(new_img[:, :, 0], cmap="Greys_r")
    ax[2].set_title("Reconstructed high-resolution depth map")
    plt.show()


def save_depth_superresolution(ref_img, target_img, new_img, file_name):
    ref_img_pil = Image.fromarray(ref_img.astype("uint8"), mode="YCbCr").convert("RGB")
    ref_img_pil.save("results/depth_superresolution_" + file_name + "_ref.png")
    target_img_pil = Image.fromarray(target_img[:, :, 0].astype("uint8"), mode="L")
    target_img_pil.save("results/depth_superresolution_" + file_name + "_target.png")
    new_img_pil = Image.fromarray(new_img[:, :, 0].astype("uint8"), mode="L")
    new_img_pil.save("results/depth_superresolution_" + file_name + "_new.png")

def ref_step_image_BW(n=100):
    ref_img = 0.7 + 0*np.random.normal(loc=0, scale=0.2, size=(n, n))
    ref_img[np.tril_indices_from(ref_img)] -= 0.4
    ref_img = (np.clip(ref_img, a_min=0, a_max=1) * 255)[:, :, None].astype(int)
    return ref_img

def step_image_BW(n=100):
    ref_img = 0.7 + np.random.normal(loc=0, scale=0.15, size=(n, n))
    ref_img[np.tril_indices_from(ref_img)] -= 0.4
    ref_img = (np.clip(ref_img, a_min=0, a_max=1) * 255)[:, :, None].astype(int)
    return ref_img


class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
    ):
        super().__init__()

        #self.edge_net = EdgesNet()

        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        # the period for visualizing training samples
        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        EDGES = True
        if EDGES:
            self.weights_x = nn.Parameter(torch.tensor([[1.0], [2.0], [1.0]], requires_grad=True, device=self.device))
            self.zero_vector = torch.zeros((3, 1), device=self.device)
            self.weights_y = nn.Parameter(torch.tensor([[1.0, 2.0, 1.0]], requires_grad=True, device=self.device))
            self.zero_y_vector = torch.zeros((1, 3), device=self.device)
        #self.conv1 = nn.Conv2d(3, 3, 3, device=self.device, padding='same')


    def forward(self, batched_inputs):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        images = self.preprocess_image(batched_inputs)
        it = images.tensor
        EDGES = True
        if EDGES:
            kernel_x = torch.cat((self.weights_x, self.zero_vector, -self.weights_x), 1)
            kernel_x = kernel_x.view(1, 1, 3, 3).to(self.device)
            # Apply edges on grayscale channel only for efficiency
            x1 = F.conv2d(it[:,[3]], kernel_x, padding='same')
            # Edges Y Direction
            kernel_y = torch.cat((self.weights_y, self.zero_y_vector, -self.weights_y), 0)
            kernel_y = kernel_y.view(1, 1, 3, 3).to(self.device)
            x2 = F.conv2d(it[:, [3]], kernel_y, padding='same')

            edge_mag = torch.sqrt(torch.pow(x1, 2) + torch.pow(x2, 2) + 1e-6)

            # for name, param in self.named_parameters():
            #     if param.requires_grad and name == "weights_y":
            #         print(name, param.data, param.grad)
            it = torch.concat((it[:, [0, 1, 2]], edge_mag), 1)
        #breakpoint()
        # DINO DETR
        if self.training:
            batch_size, num_channels, H, W = it.shape
            img_masks = it.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = it.shape
            img_masks = it.new_zeros(batch_size, H, W)

        features = self.backbone(it)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.embed_dim,
                label_enc=self.label_enc,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            # visualize training samples
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    box_cls = output["pred_logits"]
                    box_pred = output["pred_boxes"]
                    results = self.inference(box_cls, box_pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)
            
            # compute loss
            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy()
            )
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord


    def preprocess_image(self, batched_inputs, edges_only=False, gray_only=False):
        """GPU Gaussian followed by Sobel"""
        # # [Batch size, num_channels, height, width]
        adaptive_edges = True
        rgb_only = False
        bilateral = False
        bilateral_only = False

        if rgb_only:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
            return ImageList.from_tensors(images)
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        if bilateral_only:
            for index, im in enumerate(images):
                if im.device.type != "cpu":
                    im = im.cpu()
                cpu_im = np.array(im)
                cpu_im = cpu_im.reshape(cpu_im.shape[1], cpu_im.shape[2], cpu_im.shape[0])
                filtered = cv.bilateralFilter(cpu_im, 9, 75, 75)
                filtered = filtered.reshape(filtered.shape[2], filtered.shape[0], filtered.shape[1])
                images[index] = self.normalizer(torch.from_numpy(filtered).to(self.device))
            images = ImageList.from_tensors(images)
            return images
        elif adaptive_edges:
            gaussian_kernel = [[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]]
            gaussian_kernel = torch.FloatTensor([[x / 273 for x in y] for y in gaussian_kernel]).unsqueeze(0).unsqueeze(0).to(self.device)
            for index, im in enumerate(images):
                gray_im = transforms.Grayscale()(im)
                if bilateral:
                    gray_cpu = np.array(gray_im.cpu())
                    gray_cpu = gray_cpu.reshape(gray_cpu.shape[1], gray_cpu.shape[2], gray_cpu.shape[0])
                    gray_im = smoothing(ref_img=gray_cpu, lambd=10, sigma_xy=50, sigma_l=50, sigma_s=None, sigma_r=None, show=False)
                smooth_img = None
                if bilateral:
                    gray_im = gray_im.reshape(gray_im.shape[2], gray_im.shape[0], gray_im.shape[1])
                    smooth_img = torch.from_numpy(gray_im).to(self.device)
                else:
                    smooth_img = F.conv2d(gray_im, gaussian_kernel, padding='same')
                images[index] = torch.cat([images[index], smooth_img], dim=0)
            images = ImageList.from_tensors(images)
            return images
        return images


    def nms(self, bounding_boxes, confidence_score, threshold):
        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []
        # Bounding boxes
        #boxes = np.array(bounding_boxes)
        boxes = np.array(torch.clone(bounding_boxes).detach().cpu())

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        #score = confidence_score.cpu()
        score = np.array(torch.tensor(confidence_score).cpu())

        # Picked bounding boxes
        picked_boxes = []
        picked_score = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        #order = np.argsort(score)
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            #picked_index.append(index)
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]
        return picked_boxes, picked_score

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []
        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        prob = box_cls.sigmoid()
        prob_scores = [x[0] for x in prob[0]]
        boxes = box_pred[0]
        # if len(picked_boxes > self.select_box_nums_for_evaluation):
            # break

        picked_boxes, picked_scores = None, None
        for thresh in range(1, 10):
            picked_boxes, picked_scores = self.nms(boxes, prob_scores, 1 - thresh / 10)
            if len(picked_boxes) <= self.select_box_nums_for_evaluation:
                break

        # topk_values, topk_indexes = torch.topk(
        #     prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        # )
        # # highest probability score for each query detection, 300
        # scores = topk_values
        # topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        # labels = topk_indexes % box_cls.shape[2]
        
        # boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        scores = torch.tensor(picked_scores, device=self.device).clone().detach()
        scores = scores.reshape((1, len(picked_boxes)))
        labels = torch.zeros((1, len(picked_scores)), device=self.device)
        boxes = torch.stack(picked_boxes).reshape((1, len(picked_boxes), 4))
        boxes = boxes.clone().detach()

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets
