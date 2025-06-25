import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import zoom


def normalize(array: NDArray) -> NDArray:
    return array / abs(array).max()


def crop_center(
        array: NDArray,
        crop_height: int | None = None,
        crop_width: int | None = None
) -> NDArray:
    y, x = array.shape
    if crop_height is None:
        crop_height = y // 2

    if crop_width is None:
        crop_width = x // 2

    start_x = x // 2 - (crop_width // 2)
    start_y = y // 2 - (crop_height // 2)
    return array[start_y:start_y + crop_height, start_x:start_x + crop_width]


def create_grid(
        num_pixels_x: int,
        dx: float | None = None,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> tuple[NDArray, ...]:
    if dx is None:
        dx = 1

    if dy is None:
        dy = dx

    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    width = dx * num_pixels_x
    height = dy * num_pixels_y
    x = np.linspace(-width / 2, width / 2, num_pixels_x)
    y = np.linspace(-height / 2, height / 2, num_pixels_y)
    return np.meshgrid(x, y)


def create_radial_grid(
        num_pixels_x: int,
        dx: float | None = None,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> NDArray[np.float64]:
    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    if dy is None:
        dy = dx

    x, y = create_grid(num_pixels_x, dx, num_pixels_y, dy)
    return np.sqrt(x ** 2 + y ** 2)


# noinspection PyUnresolvedReferences
def zoom_matrix(input_matrix: NDArray, zoom_factor: float) -> NDArray:
    original_shape = np.array(input_matrix.shape)
    scaled_matrix = zoom(input_matrix, zoom_factor, order=3)

    new_shape = np.array(scaled_matrix.shape)
    pad_or_crop = original_shape - new_shape
    pad_width = [(max(0, int(np.floor(abs(diff) / 2))), max(0, int(np.ceil(abs(diff) / 2)))) for diff in pad_or_crop]

    if any(diff > 0 for diff in pad_or_crop):
        scaled_matrix = np.pad(scaled_matrix, pad_width)
    else:
        scaled_matrix = crop_center(scaled_matrix, *original_shape)

    return scaled_matrix[:original_shape[0], :original_shape[1]]


# noinspection PyTypeChecker
def multiply_spherical_wavefront(
        object: NDArray,
        z: float,
        dx: float,
        wavelength: float,
        xi: float = 0.0,
        eta: float = 0.0
) -> NDArray[np.complex128]:
    num_pixels_y, num_pixels_x = object.shape
    x, y = create_grid(num_pixels_x, dx, num_pixels_y)
    k = 2 * np.pi / wavelength
    r = np.sqrt((x - xi) ** 2 + (y - eta) ** 2 + z ** 2)
    return object * np.exp(1j * k * r) / r

mean_squared_error = lambda x, y: np.mean((x - y) ** 2)