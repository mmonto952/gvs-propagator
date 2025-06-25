import numpy as np
import dask
import dask.array as da
from typing import Any
from numpy.typing import NDArray

from gvs_propagator.utils import (
    normalize,
    create_grid,
    multiply_spherical_wavefront,
)
from gvs_propagator.coherent import (
    compute_angular_spectrum,
    compute_fresnel_convolution,
    compute_shifted_fresnel,
    compute_intensity_for_wavefront,
)


def compute_complex_coherence_factor(source_intensity: NDArray[np.float64]) -> NDArray[np.complex128]:
    return normalize(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(source_intensity))))


def compute_schells_propagation(
        coherent_intensity: NDArray,
        complex_coherence_factor: NDArray,
        z: float,
        wavelength: float,
) -> NDArray[np.complex128]:
    convolved_fields = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(coherent_intensity))) * complex_coherence_factor
    propagated_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(convolved_fields)))
    return np.abs(1 / (wavelength * z) ** 2 * propagated_field)


def compute_schells_theorem(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z: float,
        wavelength: float,
) -> NDArray[np.complex128]:
    return compute_schells_propagation(
        np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(object)))) ** 2,
        complex_coherence_factor,
        z,
        wavelength,
    )


def compute_gvs(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z_source_object: float,
        z_object_camera: float,
        wavelength: float,
        dx: float,
) -> NDArray[np.complex128]:
    num_pixels_y, num_pixels_x = object.shape
    x, y = create_grid(num_pixels_x, dx, num_pixels_y)
    phase_factor = np.exp(1j * np.pi / wavelength * (1 / z_object_camera + 1 / z_source_object) * (x ** 2 + y ** 2))
    return compute_schells_theorem(object * phase_factor, complex_coherence_factor, z_object_camera, wavelength)


def compute_mutual_intensity_integral(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z: float,
        wavelength: float,
        dx_camera: float,
        dx_object: float | None = None,
        chunk_size: int = 1000,
) -> NDArray:
    num_pixels_y, num_pixels_x = object.shape
    if dx_object is None:
        dx_object = dx_camera

    x_object, y_object = create_grid(num_pixels_x, dx_object, num_pixels_y)
    x_camera, y_camera = create_grid(num_pixels_x, dx_camera, num_pixels_y)

    x_camera = da.from_array(x_camera.flatten(), chunks=(chunk_size,))
    y_camera = da.from_array(y_camera.flatten(), chunks=(chunk_size,))

    xi, eta = np.where(object != 0)

    x_object = da.from_array(x_object[xi, eta], chunks=(chunk_size,))
    y_object = da.from_array(y_object[xi, eta], chunks=(chunk_size,))

    d_object_to_grid = da.sqrt(
        (x_object[:, None] - x_camera[None, :]) ** 2 +
        (y_object[:, None] - y_camera[None, :]) ** 2 + z ** 2
    )
    phase_difference = da.exp(-2j * np.pi / wavelength * (
            d_object_to_grid[:, None, :] - d_object_to_grid[None, :, :]
    ))
    obliquity_factor = z ** 2 / (
            (d_object_to_grid[:, None, :] ** 2) * (d_object_to_grid[None, :, :]) ** 2
    )

    field_contributions = (
            object[xi, eta][:, None, None] *
            np.conj(object)[xi, eta][None, :, None] *
            complex_coherence_factor[
                xi[:, None] - xi[None, :],
                eta[:, None] - eta[None, :]
            ][:, :, None] *
            phase_difference *
            obliquity_factor
    )

    intensity_flat = da.abs(da.sum(field_contributions, axis=(0, 1))) / (wavelength * z) ** 2
    return intensity_flat.reshape(num_pixels_y, num_pixels_x).compute() * dx_object ** 4


def compute_mutual_intensity_integral_cross_section(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z_source_object: float,
        z_object_camera: float,
        wavelength: float,
        dx_camera: float,
        dx_object: float | None = None,
        chunk_size: int = 1000,
) -> NDArray:
    num_pixels_y, num_pixels_x = object.shape
    if dx_object is None:
        dx_object = dx_camera

    x_object, y_object = create_grid(num_pixels_x, dx_object, num_pixels_y)
    width = dx_camera * num_pixels_x
    x_camera = np.linspace(-width / 2 - dx_camera / 2, width / 2 - dx_camera / 2, num_pixels_x)  # Shift to center
    y_camera = np.ones_like(x_camera) * x_camera[num_pixels_x // 2]
    x_camera = da.from_array(x_camera, chunks=(chunk_size,))
    y_camera = da.from_array(y_camera, chunks=(chunk_size,))

    xi, eta = np.where(object != 0)

    x_object = da.from_array(x_object[xi, eta], chunks=(chunk_size,))
    y_object = da.from_array(y_object[xi, eta], chunks=(chunk_size,))

    d_object_to_grid = da.sqrt(
        (x_object[:, None] - x_camera[None, :]) ** 2 +
        (y_object[:, None] - y_camera[None, :]) ** 2 + z_object_camera ** 2
    )
    phase_difference = da.exp(-2j * np.pi / wavelength * (
            d_object_to_grid[:, None, :] - d_object_to_grid[None, :, :]
    ))
    obliquity_factor = z_object_camera ** 2 / (
            (d_object_to_grid[:, None, :] ** 2) * (d_object_to_grid[None, :, :]) ** 2
    )
    complex_coherence_phase_factor = da.exp(-1j * np.pi / (wavelength * z_source_object) * (
            x_object[:, np.newaxis] ** 2 - x_object[np.newaxis, :] ** 2 +
            y_object[:, np.newaxis] ** 2 - y_object[np.newaxis, :] ** 2
    ))

    field_contributions = (
            object[xi, eta][:, None, None] *
            np.conj(object)[xi, eta][None, :, None] *
            complex_coherence_factor[
                xi[:, None] - xi[None, :],
                eta[:, None] - eta[None, :]
            ][:, :, None] *
            complex_coherence_phase_factor[:, :, np.newaxis] *
            phase_difference *
            obliquity_factor
    )

    intensity_flat = da.abs(da.sum(field_contributions, axis=(0, 1))) / (wavelength * z_object_camera) ** 2
    return intensity_flat.compute() * dx_object ** 4


def compute_angular_spectrum_gvs(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z_source_object: float,
        z_object_camera: float,
        dx: float,
        wavelength: float,
        **kwargs
) -> NDArray[np.complex128]:
    num_pixels_y, num_pixels_x = object.shape
    x, y = create_grid(num_pixels_x, dx, num_pixels_y)
    phase_factor = np.exp(1j * np.pi / (wavelength * z_source_object) * (x ** 2 + y ** 2))
    coherent_intensity = np.abs(
        compute_angular_spectrum(
            object * phase_factor,
            z_object_camera,
            dx,
            wavelength,
            z_source_object=z_source_object,
            **kwargs
        )
    ) ** 2
    return compute_schells_propagation(
        coherent_intensity,
        complex_coherence_factor,
        z_object_camera,
        wavelength,
    )


def compute_fresnel_convolution_gvs(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z_source_object: float,
        z_object_camera: float,
        dx: float,
        wavelength: float,
) -> NDArray[np.complex128]:
    num_pixels_y, num_pixels_x = object.shape
    x, y = create_grid(num_pixels_x, dx, num_pixels_y)
    phase_factor = np.exp(1j * np.pi / (wavelength * z_source_object) * (x ** 2 + y ** 2))
    coherent_intensity = np.abs(
        compute_fresnel_convolution(
            object * phase_factor,
            z_object_camera,
            dx, wavelength,
            z_source_object=z_source_object
        )
    ) ** 2
    return compute_schells_propagation(
        coherent_intensity,
        complex_coherence_factor,
        z_object_camera,
        wavelength,
    )


def compute_shifted_fresnel_gvs(
        object: NDArray,
        complex_coherence_factor: NDArray,
        z_source_object: float,
        z_object_camera: float,
        wavelength: float,
        dx_camera: float,
        dx_object: float | None = None,
        **kwargs,
) -> NDArray[np.complex128]:
    if dx_object is None:
        dx_object = dx_camera

    num_pixels_y, num_pixels_x = object.shape
    x, y = create_grid(num_pixels_x, dx_object, num_pixels_y, dx_object)
    phase_factor = np.exp(1j * np.pi / (wavelength * z_source_object) * (x ** 2 + y ** 2))
    coherent_intensity = np.abs(
        compute_shifted_fresnel(
            object * phase_factor,
            z_object_camera,
            wavelength,
            dx_camera,
            dx_object,
            **kwargs
        )
    ) ** 2
    return compute_schells_propagation(
        coherent_intensity,
        complex_coherence_factor,
        z_object_camera,
        wavelength,
    )


def compute_intensity_point_sources(
        object: NDArray[np.complex128],
        source_intensity: NDArray[np.float64],
        dx_source: float,
        dx_camera: float,
        z_source_object: float,
        z_object_camera: float,
        wavelength: float,
        sample_size: int = 10000
) -> NDArray[np.float64]:
    num_pixels_y, num_pixels_x = source_intensity.shape
    non_zero_pixels = list(zip(*np.where(source_intensity > 0.0)))
    if len(non_zero_pixels) > sample_size:
        indices = np.random.choice(len(non_zero_pixels), size=sample_size, replace=False)
    else:
        indices = np.arange(len(non_zero_pixels))

    intensity_point_sources = np.zeros_like(source_intensity)
    for i in indices:
        xi = (non_zero_pixels[i][1] - num_pixels_x) * dx_source
        eta = (non_zero_pixels[i][0] - num_pixels_y) * dx_source

        intensity_point_sources += abs(compute_angular_spectrum(
            multiply_spherical_wavefront(object, z_source_object, dx_camera, wavelength, xi, eta),
            z_object_camera,
            dx_camera,
            wavelength,
        )) ** 2

    return intensity_point_sources


def compute_intensity_for_source(
        xi: float,
        eta: float,
        object: NDArray[np.complex128],
        z_source_object: float,
        dx_camera: float,
        dx_object: float,
        wavelength: float,
        z_object_camera: float
):
    return abs(
        compute_angular_spectrum(
            multiply_spherical_wavefront(
                object,
                z_source_object,
                dx_object,
                wavelength,
                xi,
                eta
            ),
            z_object_camera,
            dx_camera,
            wavelength,
        )
    ) ** 2


def compute_intensity_point_sources_dask(
        object: NDArray[np.complex128],
        source_intensity: NDArray[np.float64],
        dx_source: float,
        dx_camera: float,
        z_source_object: float,
        z_object_camera: float,
        wavelength: float,
        dx_object: float | None = None,
        sample_size: int = 100,
        **kwargs: Any
) -> NDArray[np.float64]:
    if dx_object is None:
        dx_object = dx_camera

    num_pixels = source_intensity.shape[0]
    delayed_tasks = []
    non_zero_pixels = list(zip(*np.where(source_intensity > 0.0)))
    if len(non_zero_pixels) > sample_size:
        indices = np.random.choice(len(non_zero_pixels), size=sample_size, replace=False)
    else:
        indices = np.arange(len(non_zero_pixels))

    for i in indices:
        xi = (non_zero_pixels[i][0] - num_pixels // 2) * dx_source
        eta = (non_zero_pixels[i][1] - num_pixels // 2) * dx_source

        delayed_task = dask.delayed(compute_intensity_for_source)(
            xi, eta,
            object, z_source_object,
            dx_camera, dx_object, wavelength,
            z_object_camera,
            **kwargs
        )
        delayed_tasks.append(delayed_task)

    return dask.delayed(sum)(delayed_tasks).compute()


def generate_filtered_wavefronts(
        sample_size: int,
        num_pixels: int,
        sigma: float,
        source_intensity: NDArray[np.float64]
) -> NDArray[np.complex128]:
    waves = (
            np.random.normal(size=(sample_size, num_pixels, num_pixels)) + 
            1j * np.random.normal(size=(sample_size, num_pixels, num_pixels))
    )
    waves *= sigma
    return np.array([apply_pupil_filter(w, source_intensity) for w in waves])


def apply_pupil_filter(wavefront: NDArray, source_intensity: NDArray) -> NDArray[np.complex128]:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        source_intensity * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wavefront)))
    )))


def compute_intensity_cpr(
        source_intensity: NDArray[np.float64],
        dx_camera: float,
        object: NDArray[np.complex128],
        z_object_camera: float,
        wavelength: float,
        sample_size: int = 100
) -> NDArray[np.float64]:
    num_pixels = source_intensity.shape[0]
    wavefronts = generate_filtered_wavefronts(sample_size, num_pixels, 1, source_intensity)
    intensity = np.zeros_like(source_intensity)
    for wavefront in wavefronts:
        intensity += compute_intensity_for_wavefront(
            object, wavefront, z_object_camera, dx_camera, wavelength
        )

    return intensity
