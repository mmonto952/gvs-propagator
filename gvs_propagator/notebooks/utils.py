import numpy as np
import plotly.express as px
from scipy.ndimage import zoom
from plotly.graph_objs import Figure
from IPython.display import Image
from typing import Any
from numpy.typing import NDArray


def build_figure(
        image: NDArray,
        show_scale: bool = False,
        show_ticks: bool = False,
        zoom_range: tuple[float, float] | None = None,
        **kwargs: Any
) -> Figure:
    margin_top = 0
    if "title" in kwargs:
        margin_top = 50

    if "x" and "y" in kwargs:
        show_ticks = True

    fig = px.imshow(
        image,
        color_continuous_scale="gray",
        binary_string=False if show_scale else True,
        **kwargs
    )
    fig.update_layout(
        coloraxis_showscale=show_scale,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=margin_top, b=0),
        xaxis=dict(range=zoom_range),
        yaxis=dict(range=zoom_range),
    )

    if not show_ticks:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    return fig


def display_image(
        field: np.ndarray | list[np.ndarray],
        log: bool = False,
        convert_to: str | None = "amplitude",
        frame_duration: int = 100,
        interactive: bool = False,
        normalize_all_frames: bool = False,
        return_figure: bool = False,
        show_as_grid: bool = False,
        show_animation: bool = False,
        grid_titles: list[str] | None = None,
        propagation_distances: list[float] | None = None,
        image_size: int | None = None,
        **kwargs: Any
) -> Image | Figure | None:
    if type(field) != list:
        field = [field]

    match convert_to:
        case "intensity":
            images = [abs(f) ** 2 for f in field]
        case "amplitude":
            images = [abs(f) for f in field]
        case "phase":
            images = [np.angle(f) for f in field]
        case None:
            images = field
        case _:
            raise ValueError(f"Unknown conversion type: {convert_to}")

    if image_size:
        if type(images[0]) == list:
            new_images = []
            for image_list in images:
                new_images.append([zoom(image, image_size / image.shape[0]) for image in image_list])
            images = new_images
        else:
            images = [zoom(image, image_size / image.shape[0]) for image in images]

        if "x" in kwargs:
            kwargs["x"] = zoom(kwargs["x"], image_size / kwargs["x"].shape[0])
        if "y" in kwargs:
            kwargs["y"] = zoom(kwargs["y"], image_size / kwargs["y"].shape[0])

    if log:
        images = [np.log(np.where(image == 0, 1e-10, image)) for image in images]

    if normalize_all_frames:
        images = [i / i.max() for i in images]

    if len(images) == 1:
        fig = build_figure(images[0], **kwargs)
    elif show_as_grid:
        if show_animation:
            interactive = True
            fig = build_figure(
                np.array(images),
                animation_frame=0,
                facet_col=1,
                **{"title": "", "height": 512, **kwargs}
            )
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"] = {"duration": frame_duration, "redraw": True}
        else:
            fig = build_figure(np.array(images), facet_col=0, **{"title": "", "height": 300, **kwargs})

        if grid_titles:
            for i, title in enumerate(grid_titles):
                fig.layout.annotations[i]["text"] = title
    else:
        interactive = True
        fig = build_figure(np.array(images), animation_frame=0, **{"width": 512, "height": 768, **kwargs})
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"] = {"duration": frame_duration, "redraw": True}

    if propagation_distances:
        fig.update_layout(
            sliders=[
                {
                    "steps": [
                        {"label": (str(round(z, 3)))}
                        for z in propagation_distances
                    ],
                    "currentvalue": {
                        "prefix": "Propagation distance: ",
                        "suffix": " m",
                    }
                }
            ]
        )

    if return_figure:
        return fig
    elif interactive:
        fig.show()
    else:
        return Image(fig.to_image(
            width=kwargs.get("width", None),
            height=kwargs.get("height", None))
        )