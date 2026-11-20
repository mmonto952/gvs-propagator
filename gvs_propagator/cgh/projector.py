import torch
from torch.nn.functional import mse_loss, interpolate
from torchmetrics.functional.image import (
    structural_similarity_index_measure as ssim,
)
from torchvision.io import read_image, ImageReadMode
from tqdm.auto import trange

from gvs_propagator.cgh.rgb_asm import compute_angular_spectrum
from gvs_propagator.constants import nm, um, mm
from gvs_propagator.utils import normalize, crop_center


def zero_pad(tensor):
    pad_width = [tensor.shape[1] // 2, tensor.shape[1] // 2, tensor.shape[0] // 2, tensor.shape[0] // 2]
    return torch.nn.functional.pad(tensor, pad_width)


def create_grid(
        num_pixels_x: int,
        dx: float | None = None,
        num_pixels_y: int | None = None,
        dy: float | None = None,
) -> tuple[torch.Tensor, ...]:
    if dx is None:
        dx = 1

    if dy is None:
        dy = dx

    if num_pixels_y is None:
        num_pixels_y = num_pixels_x

    width = dx * num_pixels_x
    height = dy * num_pixels_y
    x = torch.linspace(-width / 2, width / 2, num_pixels_x)
    y = torch.linspace(-height / 2, height / 2, num_pixels_y)
    return torch.meshgrid(x, y, indexing="xy")


def resize(image, nx: int, ny: int, **kwargs):
    return interpolate(
        image[torch.newaxis, torch.newaxis],
        size=(ny, nx, *image.shape[2:]),
        **kwargs,
    )[0, 0]


# noinspection PyTypeChecker
def lsa_propagate(
        phase,
        nx,
        ny,
        dx,
        dy,
        dx_projection,
        z,
        wavelength,
        intensity_modulation=None,
        epsilon=1e-12
):
    if intensity_modulation is None:
        intensity_modulation = torch.ones((ny, nx))

    k = 2 * torch.pi / wavelength
    x, y = create_grid(nx, dx, ny, dy)
    r2 = x ** 2 + y ** 2

    fx = torch.arange(-nx / 2, nx / 2) / (nx * dx)
    fy = torch.arange(-ny / 2, ny / 2) / (ny * dy)
    fx, fy = torch.meshgrid(fx, fy, indexing="xy")

    laser_intensity = normalize(abs(torch.exp(-r2 / (2 * dx_projection ** 2))) ** 2)
    fresnel_transfer_function = torch.fft.ifftshift(
        torch.exp(1j * k * z - 1j * torch.pi * wavelength * z * (fx ** 2 + fy ** 2))
    )
    a_eff = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(intensity_modulation) * torch.fft.fft2(laser_intensity)))
    baseline_intensity = torch.abs(
        torch.fft.ifft2(torch.fft.fft2(phase * torch.sqrt(a_eff + epsilon)) * fresnel_transfer_function)
    ) ** 2
    far_field_analytical_beam = normalize(
        torch.exp(-4 * torch.pi ** 2 * dx_projection ** 2 * r2 / (wavelength * z) ** 2)
    )
    intensity = normalize(
        abs(torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.fft2(baseline_intensity) * torch.fft.fft2(far_field_analytical_beam))
        ))
    )
    return intensity


class CGHOptimizer:
    """Shared gradient-descent loop for the hologram solvers.

    Subclasses implement `propagate` and may override `parameters`,
    `compute_loss`, `after_step` and `format_result`.
    """

    def __init__(
            self,
            image_path,
            nx,
            ny,
            dx_slm,
            propagation_distances,
            wavelength,
            lr=0.01,
            iterations=500,
            batch_size=1,
            device="cpu",
    ):
        self.nx = nx
        self.ny = ny
        self.dx_slm = dx_slm
        self.propagation_distances = propagation_distances
        self.wavelength = wavelength
        self.lr = lr
        self.iterations = iterations
        self.batch_size = batch_size
        self.device = device
        self.image_paths = [image_path]

        self.phase_guess = torch.rand((ny, nx), requires_grad=True, device=device)

    def parameters(self):
        return [self.phase_guess]

    def propagate(self, z):
        raise NotImplementedError

    def compute_loss(self, result, target_image):
        # noinspection PyTypeChecker
        return 1 - ssim(
            result[torch.newaxis, torch.newaxis],
            target_image[torch.newaxis, torch.newaxis]
        ) + mse_loss(result, target_image)

    def after_step(self):
        pass

    def format_result(self, best_parameters):
        (best_guess,) = best_parameters
        return torch.exp(2j * torch.pi * best_guess)

    def solve(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        best_loss = float("inf")
        with trange(self.iterations) as t:
            for _ in t:
                optimizer.zero_grad()
                all_losses = []
                losses = []
                for j in range(len(self.image_paths)):
                    z = self.propagation_distances[j]
                    result = self.propagate(z)

                    target_image = (read_image(self.image_paths[j], mode=ImageReadMode.GRAY)[0] / 255).to(self.device)
                    loss = self.compute_loss(result, target_image)
                    losses.append(loss)

                    if len(losses) % self.batch_size == 0 or j + 1 == len(self.image_paths):
                        loss = sum(losses) / len(losses)
                        all_losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        self.after_step()
                        optimizer.zero_grad()
                        losses = []

                loss = sum(all_losses) / len(all_losses)
                if loss < best_loss:
                    best_loss = loss
                    best_parameters = [parameter.clone() for parameter in self.parameters()]

                t.set_postfix(loss=loss)

        return self.format_result(best_parameters)


class CoherentOptimizer(CGHOptimizer):
    def propagate(self, z):
        phase = zero_pad(torch.exp(2j * torch.pi * self.phase_guess))
        return normalize(crop_center(
            abs(compute_angular_spectrum(phase, z, self.dx_slm, self.wavelength, device=self.device))
        ))

    def compute_loss(self, result, target_image):
        return super().compute_loss(normalize(result), normalize(target_image))


class PhaseOnlyOptimizer(CGHOptimizer):
    def __init__(
            self,
            image_path,
            nx,
            ny,
            dx_slm,
            dx_projection,
            propagation_distances,
            wavelength,
            lr=0.01,
            iterations=500,
            batch_size=1,
            device="cpu",
    ):
        super().__init__(
            image_path,
            nx,
            ny,
            dx_slm,
            propagation_distances,
            wavelength,
            lr=lr,
            iterations=iterations,
            batch_size=batch_size,
            device=device,
        )
        self.dx_projection = dx_projection

    def intensity_modulation(self):
        return None

    def propagate(self, z):
        intensity_modulation = self.intensity_modulation()
        phase = zero_pad(torch.exp(2j * torch.pi * self.phase_guess))
        return crop_center(lsa_propagate(
            phase,
            self.nx * 2,
            self.ny * 2,
            self.dx_slm,
            self.dx_slm,
            self.dx_projection,
            z,
            self.wavelength,
            intensity_modulation=intensity_modulation,
        ))


class IntensityPhaseOptimizer(PhaseOnlyOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nx_intensity = round(self.nx * self.dx_slm / self.dx_projection)
        ny_intensity = round(self.ny * self.dx_slm / self.dx_projection)
        self.intensity_modulation_guess = torch.ones(
            (ny_intensity, nx_intensity), requires_grad=True, device=self.device
        )

    def parameters(self):
        return [self.phase_guess, self.intensity_modulation_guess]

    def intensity_modulation(self):
        return zero_pad(resize(self.intensity_modulation_guess, self.nx, self.ny))

    def after_step(self):
        self.intensity_modulation_guess.data.clamp_(min=0, max=1)

    def format_result(self, best_parameters):
        best_phase_guess, best_intensity_guess = best_parameters
        return (
            best_intensity_guess,
            torch.exp(2j * torch.pi * best_phase_guess)
        )
