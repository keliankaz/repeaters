# SAFOD Repeaters Analysis

This project analyzes repeating earthquakes (repeaters) in the creeping section of the San Andreas Fault. The analyses are presented in the following notebooks: 

- `SAFOD_repeaters.ipynb`: Analysis of microseismicity near the SAFOD (Fig. 3).
- `manuscript_figures.ipynb`: Analysis of all repeaing earthquakes in the catalog along the creeping section (Figs. 2 and S1).
- `subduction_zone.ipynb`: Analysis of neiboring subduction zone earthquakes (Fig. 4)

## Dependencies

The dependencies are listed in the `requirements.yml` file. To install the dependencies, run the following command:

```bash
conda env create -f requirements.yml
conda activate repeaters
```

## Dataset

The analysis uses an updated catalog for the creeping section of the San Andreas Fault. The catalog preparation details are described in:

Y. Li, R. Bürgmann, T. Taira, Spatiotemporal Variations of Surface Deformation, Shallow Creep Rate, and Slip Partitioning Between the San Andreas and Southern Calaveras Fault. J. Geophys. Res. Solid Earth 128, e2022JB025363 (2023).

The historical record of subduction zone earthquakes is from:

B. Philibosian, A. J. Meltzner, Segmentation and supercycles: A catalog of earthquake rupture patterns from the Sumatran Sunda Megathrust and other well-studied faults worldwide. Quat. Sci. Rev. 241, 106390 (2020).


