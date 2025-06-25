import numpy as np
from numpy.typing import NDArray

from gvs_propagator.utils import (
    create_grid,
    create_radial_grid,
)


def create_circular_mask(
        num_pixels_x: int,
        dx: float,
        radius: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> NDArray[np.float64]:
    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    if dy is None:
        dy = dx

    r = create_radial_grid(num_pixels_x, dx, num_pixels_y, dy)
    source_intensity = np.zeros_like(r)
    source_intensity[r < radius] = 1.0

    return source_intensity


def create_double_pinhole_mask(
        num_pixels_x: int,
        dx: float,
        radius: float,
        separation: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> NDArray[np.float64]:
    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    if dy is None:
        dy = dx

    pixels_from_axis = int(separation / (2 * dx))
    pinhole = create_circular_mask(num_pixels_x, dx, radius, num_pixels_y, dy)
    object = np.roll(pinhole, -pixels_from_axis) + np.roll(pinhole, pixels_from_axis)
    return np.pad(object, ((num_pixels_y // 2,) * 2, (num_pixels_x // 2,) * 2))


def create_slit_mask(
        num_pixels_x: int,
        dx: float,
        width: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
        half_height: float = np.inf,
) -> NDArray[np.float64]:
    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    if dy is None:
        dy = dx

    x, y = create_grid(num_pixels_x, dx, num_pixels_y, dy)
    slit = np.zeros_like(x)
    slit[(abs(x) < width / 2) & (-half_height < y) & (y < half_height)] = 1.0
    return slit


def create_double_slit_mask(
        num_pixels_x: int,
        dx: float,
        width: float,
        separation: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
        **kwargs
) -> NDArray[np.float64]:
    pixels_from_axis = int(separation / (2 * dx))
    slit = create_slit_mask(num_pixels_x * 2, dx, width, num_pixels_y * 2, dy, **kwargs)
    return np.roll(slit, -pixels_from_axis) + np.roll(slit, pixels_from_axis)


# noinspection PyTypeChecker
def create_triple_slit_mask(
        num_pixels_x: int,
        dx: float,
        width: float,
        separation: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
        **kwargs
) -> NDArray[np.float64]:
    pixels_from_axis = int(separation / (2 * dx))
    pixels_width = width // dx
    slit = create_slit_mask(num_pixels_x * 2, dx, width, num_pixels_y * 2, dy, **kwargs)
    return slit + np.roll(slit, -pixels_from_axis - pixels_width) + np.roll(slit, pixels_from_axis + pixels_width)


# noinspection PyTypeChecker
def create_quadruple_slit_mask(
        num_pixels_x: int,
        dx: float,
        width: float,
        separation: float,
        num_pixels_y: int | None = None,
        dy: float | None = None,
        **kwargs
) -> NDArray[np.float64]:
    pixels_from_axis = int(separation / (2 * dx))
    pixels_width = width // dx
    slit = create_slit_mask(num_pixels_x * 2, dx, width, num_pixels_y * 2, dy, **kwargs)
    object = np.roll(slit, -pixels_from_axis) + np.roll(slit, pixels_from_axis)
    return np.roll(object, -pixels_from_axis - pixels_width) + np.roll(object, pixels_from_axis + pixels_width)
