# Generalized Van Cittert-Zernike Schell propagator

![USAF animation](assets/usaf.gif)

This is a simplified repository showcasing different functionalities of the
GVS propagator, which is an efficient algorithm for simulating partially
coherent light. The model uses the Van Cittert-Zernike theorem to find the
complex coherence factor from a light source reaching the object plane,
and it then uses the generalized Schell’s theorem to compute the diffraction
pattern produced at a given propagation distance, not limited to the far field.


# CGH using the GVS propagator

The repository also contains an example of how to use the GVS propagator to
simulate a computer-generated hologram using Pytorch.

https://github.com/user-attachments/assets/723ef206-0f53-45b7-ad6f-021dcce40fff


# Partially Coherent CGH using a Scanning Laser Projector

This repository also contains the implementation of a partially spatially
coherent propagation algorithm based on GVS, modified to efficiently model
a projector's light-scanning process.

![LSA vs Exhaustive](assets/lsa-vs-exhaustive.png)

# 🗂️ Repository Structure

These are the files to take a look at 👀:

- `gvs_propagator/`
  - `coherent.py`: Different coherent propagation algorithms.
  - `partially_coherent.py`: Implementation of partially coherent propagation algorithms.
  - `cgh/`
    - `rgb_asm.py`: Application of GVS to computer-generated holography using Pytorch. 
    - `projector.py`: Partially coherent CGH using a scanning laser projector.
  - `notebooks/`
    - `gvs_propagator.ipynb`: Main example notebook showcasing the proposed algorithms.
    - `cgh.ipynb`: Example notebook showing a CGH optimization using GVS and its results.
    - `projector.ipynb`: Example notebook showing CGH optimizations using a scanning laser projector.


- 🚀 Take a look at the exported HTML version of the [GVS propagator notebook](https://mmonto952.github.io/gvs-propagator/).
- 🚀 Take a look at the exported HTML version of the [CGH using GVS notebook](https://mmonto952.github.io/partially-coherent-cgh/).
- 🚀 Take a look at the exported HTML version of the [Scanning Laser Projector notebook](https://mmonto952.github.io/projector/).

# 📦 Required Packages

This project uses [Poetry](https://python-poetry.org/) for dependency management. The main dependencies are:

- Python 3.11+
- numpy
- scipy
- dask
- pandas
- plotly
- imageio
- kaleido (for static image export with plotly)
- pytorch
- torchmetrics

# 📖 Citation

Bibtex entry for [GVS](https://opg.optica.org/oe/fulltext.cfm?uri=oe-33-22-46313):

```text
@article{Montoya2025GeneralizedLight,
    title = {{Generalized Van Cittert-Zernike Schell propagator: an efficient algorithm for simulating partially coherent light}},
    year = {2025},
    journal = {Optics Express},
    author = {Montoya, Manuel and Lopera, Maria J. and Nie, Yunfeng and Blinder, David},
    number = {22},
    month = {11},
    pages = {46313--46325},
    volume = {33},
    publisher = {Optica Publishing Group},
    doi = {10.1364/OE.571673},
    issn = {1094-4087},
}
```

Bibtex entry for [CGH using GVS](https://opg.optica.org/ol/fulltext.cfm?uri=ol-51-8-2084):

```text
@article{Montoya2026Computer-generatedPropagator,
    title = {{Computer-generated holography using the generalized Van Cittert-Zernike Schell propagator}},
    year = {2026},
    journal = {Optics Letters},
    author = {Montoya, Manuel and Nie, Yunfeng and Blinder, David},
    number = {8},
    month = {4},
    pages = {2084--2087},
    volume = {51},
    publisher = {Optica Publishing Group},
    doi = {10.1364/OL.595741},
    issn = {1539-4794},
}
```

The citation for *Partially coherent computer-generated holography using a scanning laser projector* will be available once the article has been published.
