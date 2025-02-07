# CA_Trees.py

## Overview
`CA_Trees.py` is a Python implementation of a **Cellular Automata (CA) simulation model** used to analyze the impact of **urban tree placement** on air quality in Amsterdam. The model simulates interactions between key city components (trees, roads, vehicles, buildings, and water).

## Project Context
This script is part of a broader project, which includes:
- **Cellular Automata Simulation**: Modeling car pollution diffusion and tree absorption.
- **Plotly Dashboard**: An interactive tool for visualizing and experimenting with tree placements.
- **UI/UX Validation**: A qualitative study evaluating dashboard usability.

## Key Features
- **Grid Mapping**: Represents Amsterdam as a grid of interacting actors.
- **PM Diffusion Modeling**: Uses Fick's laws to simulate air pollutant spread.
- **Tree Absorption Simulation**: Implements Nowak's formula to calculate tree efficiency in reducing PM concentrations.
- **Integration with Plotly Dashboard**: Outputs simulation results for visualization.

## Related Files
- **Data_Preprocessing.ipynb** – Prepares raw data for simulation input.
- **CA_Trees.py** – Main Cellular Automata simulation script.
- **DSP_DashAPP.ipynb** – Jupyter Notebook for dashboard implementation.


## Dependencies
	•	Python 3.x
	•	NumPy, Matplotlib, Shapely, GeoPandas
	•	Dash, Plotly (for dashboard integration)
