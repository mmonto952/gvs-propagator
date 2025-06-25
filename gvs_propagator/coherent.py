import numpy as np
from numpy.typing import NDArray

from gvs_propagator.utils import create_grid, multiply_spherical_wavefront


def compute_angular_spectrum(
        object: NDArray,
        z_object_camera: float,
        dx: float,
        wavelength: float,
        spherical: bool = False,
        z_source_object: float = 0.0,
        **kwargs
) -> NDArray[np.complex128]:
    def quadratic_position_index(n: int) -> NDArray[np.float64]:
        return (wavelength / dx / n * np.arange(-n / 2, n / 2)) ** 2
    
    if spherical:
        object = multiply_spherical_wavefront(object, z_source_object, dx, wavelength, **kwargs)

    qx = quadratic_position_index(object.shape[1])
    qy = quadratic_position_index(object.shape[0])
    qy = qy[:, np.newaxis]
    q = qx + qy

    if dx < wavelength / np.sqrt(2):
        condition = q > 1.0
        q[condition] = 1.0
        propagation_kernel = np.exp(2j * np.pi / wavelength * z_object_camera * np.sqrt(1 - q))
        propagation_kernel[condition] = 0.0

    else:
        propagation_kernel = np.exp(2j * np.pi / wavelength * z_object_camera * np.sqrt(1 - q))

    return np.fft.ifft2(np.fft.fft2(object) * np.fft.ifftshift(propagation_kernel))


def compute_shifted_fresnel(
        object: NDArray,
        z: float,
        wavelength: float,
        dx_camera: float,
        dx_object: float | None = None,
        shift_camera_x: float = 0.0,
        shift_camera_y: float = 0.0,
        shift_object_x: float = 0.0,
        shift_object_y: float = 0.0,
) -> NDArray[np.complex128]:
    num_pixels_y, num_pixels_x = object.shape
    k = 2 * np.pi / wavelength

    if dx_object is None:
        dx_object = dx_camera

    m, n = create_grid(num_pixels_x, 1, num_pixels_y)
    x_object, y_object = m * dx_object + shift_object_x, n * dx_object + shift_object_y
    x_camera, y_camera = m * dx_camera + shift_camera_x, n * dx_camera + shift_camera_y
    s = dx_object * dx_camera / (wavelength * z)

    constant_phase_factor = (
            np.exp(1j * k * z) /
            (1j * wavelength * z) *
            np.exp(1j * k / (2 * z) * (x_camera ** 2 + y_camera ** 2)) *
                np.exp(-1j * k / z * (shift_object_x * m * dx_object + shift_object_y * n * dx_object)) *
            np.exp(-1j * np.pi * (s * m ** 2 + s * n ** 2))
    )
    object_phase_factor = (
            np.exp(1j * k / (2 * z) * (x_object ** 2 + y_object ** 2)) *
            np.exp(-2j * np.pi * (m * s * shift_camera_x + n * s * shift_camera_y)) *
            np.exp(-1j * np.pi * (s * m ** 2 + s * n ** 2))
    )

    propagation_phase_factor = np.exp(1j * np.pi * (s * m ** 2 + s * n ** 2))
    object_fft = np.fft.fft2(object * object_phase_factor)
    propagation_phase_factor_fft = np.fft.fft2(propagation_phase_factor)
    return (
            constant_phase_factor *
            np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(object_fft * propagation_phase_factor_fft)))
    )


def compute_fresnel_convolution(
        object: NDArray,
        z: float,
        dx: float,
        wavelength: float,
        spherical: bool = False,
        z_source_object: float = 0.0
) -> NDArray[np.complex128]:
    def quadratic_position_index(n):
        return (wavelength / dx / n * np.arange(-n / 2, n / 2)) ** 2

    if spherical:
        object = multiply_spherical_wavefront(object, z_source_object, dx, wavelength)

    num_pixels_y, num_pixels_x = object.shape
    phase_factor = np.exp(
        1j * np.pi * z / wavelength *
        (2 - quadratic_position_index(num_pixels_x)[np.newaxis, :]
         - quadratic_position_index(num_pixels_y)[:, np.newaxis])
    )
    return np.fft.ifft2(np.fft.fft2(object) * np.fft.ifftshift(phase_factor))


def compute_intensity_for_wavefront(
        object: NDArray,
        wavefront: NDArray,
        z_object_camera: float,
        dx_camera: float,
        wavelength: float
) -> NDArray[np.float64]:
    return np.abs(
        compute_angular_spectrum(
            object * wavefront,
            z_object_camera,
            dx_camera,
            wavelength,
        )
    ) ** 2