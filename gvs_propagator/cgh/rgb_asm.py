import numpy as np
import torch

from torch.nn.functional import mse_loss, interpolate
from torchvision.transforms import CenterCrop
from torchvision.io import read_image
from tqdm.auto import trange

from gvs_propagator.utils import normalize


def compute_complex_coherence_factor(source_intensity: torch.Tensor) -> torch.Tensor:
    return normalize(torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(source_intensity))))


def compute_angular_spectrum(
        field: torch.Tensor, z: float, dx: float, wavelength: float, device: str = "cpu",
) -> torch.Tensor:
    def quadratic_position_index(n: int):
        return (wavelength / dx / n * torch.arange(-n / 2, n / 2, device=device)) ** 2

    qx = quadratic_position_index(field.shape[1])
    qy = quadratic_position_index(field.shape[0])
    qy = qy[:, torch.newaxis]
    propagation_kernel = torch.exp(2j * torch.pi / wavelength * z * torch.sqrt(1 - qx - qy))
    return torch.fft.ifft2(torch.fft.fft2(field) * torch.fft.ifftshift(propagation_kernel))


def forward_model(
        source_guess: torch.Tensor,
        object_phase: torch.Tensor,
        phase_factor: torch.Tensor,
        nx_source_numerical: int,
        z_object_camera: float,
        dx: float,
        wavelength: float,
        reference_wavelength: float,
        device: str = "cpu",
):
    source_guess_interpolated = interpolate(
        source_guess[torch.newaxis, torch.newaxis],
        size=(nx_source_numerical, nx_source_numerical),
        mode="bilinear",
    )[0, 0]
    complex_coherence_factor = compute_complex_coherence_factor(
        CenterCrop([nx_source_numerical, nx_source_numerical])(source_guess_interpolated)
    )
    real = interpolate(
        torch.real(complex_coherence_factor)[None, None],
        size=object_phase.size(),
        mode="bilinear",
    )[0, 0]
    imag = interpolate(
        torch.imag(complex_coherence_factor)[None, None],
        size=object_phase.size(),
        mode="bilinear",
    )[0, 0]

    object = torch.exp(2j * torch.pi * object_phase * wavelength / reference_wavelength) * phase_factor
    object_spectral_density = torch.abs(
        compute_angular_spectrum(object, z_object_camera, dx, wavelength, device)
    ) ** 2
    object_autocorrelation = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(object_spectral_density)))

    convolved_fields = object_autocorrelation * torch.complex(real, imag)
    propagated_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(convolved_fields)))
    return torch.abs(propagated_field)


# noinspection PyUnresolvedReferences
def solve_inverse_problem(
        image_paths,
        nx_source,
        nx_source_numerical,
        phase_factors,
        propagation_distances,
        dx,
        wavelengths,
        reference_wavelength,
        nx_slm,
        ny_slm,
        nx_camera,
        ny_camera,
        lr=0.01,
        iterations=500,
        device="cpu",
        batch_size=1,
):
    pad_or_crop_slm = CenterCrop([ny_slm * 2, nx_slm * 2])
    crop_camera = CenterCrop([ny_camera, nx_camera])
    phase_factors = [torch.tensor(p, device=device, dtype=torch.complex64) for p in phase_factors]

    source_guess_red = torch.rand((nx_source, nx_source), requires_grad=True, device=device)
    source_guess_green = torch.rand_like(source_guess_red, requires_grad=True, device=device)
    source_guess_blue = torch.rand_like(source_guess_red, requires_grad=True, device=device)
    source_guess = [source_guess_red, source_guess_green, source_guess_blue]
    phase_guess = torch.zeros((ny_slm, nx_slm), requires_grad=True, device=device)

    optimizer = torch.optim.Adam([source_guess_red, source_guess_green, source_guess_blue, phase_guess], lr=lr)

    best_loss = float("inf")
    best_source_guess = None
    best_phase_guess = None
    images = [read_image(i) for i in image_paths]
    with trange(iterations) as t:
        for _ in t:
            optimizer.zero_grad()
            all_losses = []
            losses = []
            max_intensities = []
            for j in range(len(images)):
                image = images[j]
                for k in range(len(source_guess)):
                    result = forward_model(
                        source_guess[k],
                        pad_or_crop_slm(phase_guess),
                        phase_factors[k],
                        nx_source_numerical[j],
                        propagation_distances[j],
                        dx,
                        wavelengths[k],
                        reference_wavelength,
                        device=device,
                    )
                    if j == 0:
                        with torch.no_grad():
                            max_intensities.append(result.max())
                    result = crop_camera(result / max_intensities[k])

                    diffraction_target = image[k].to(device) / 255.0
                    losses.append(mse_loss(result, diffraction_target))

                if len(losses) == batch_size * 3 or j + 1 == len(images):
                    loss = sum(losses) / len(losses)
                    all_losses.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    source_guess_red.data.clamp_(min=0, max=1)
                    source_guess_green.data.clamp_(min=0, max=1)
                    source_guess_blue.data.clamp_(min=0, max=1)
                    optimizer.zero_grad()

            loss = sum(all_losses) / len(all_losses)
            if loss < best_loss:
                best_loss = loss
                best_source_guess = [source_guess_red.clone(), source_guess_green.clone(), source_guess_blue.clone()]
                best_phase_guess = phase_guess.clone()

            t.set_postfix(loss=loss)

    best_result_source = np.moveaxis(np.array(
        [s.detach().cpu().numpy() for s in best_source_guess]
    ), 0, -1)
    best_result_phase = best_phase_guess.detach().cpu().numpy()
    return best_result_source, best_result_phase
