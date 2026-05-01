Plots
=====

This section provides an overview of the plotting capabilities available in pythermalcomfort.
The library offers class-based utilities to visualize model results with Matplotlib.
We also provide examples of how to create visually compelling and informative plots using popular Python libraries such as Matplotlib and Seaborn.

To use the plotting functions we included in pythermalcomfort, please install the optional dependencies using:

.. code-block:: bash

    pip install pythermalcomfort[plots]

Typical Use Cases
-----------------

- Comparing the results of different thermal comfort models.
- Visualizing the impact of various parameters (e.g., temperature, humidity, clothing insulation) on model output.
- Creating custom plots to communicate findings effectively.
- Analyse the comfort levels in different environments.
- Process, analyze, and visualise:
    - climate data in different formats (e.g., CSV, Excel, JSON, EPW)
    - results from building simulation software.
    - results from sensor data.

Plotting Examples
-----------------

The `pythermalcomfort.plots.matplotlib` module includes configurable plotting classes.
These classes can be used to compare model results and to visualize how variables affect outcomes.

The `plot` module has a sub-module for different backends, currently supporting Matplotlib only, but we plan to add support for other libraries in the future (e.g., Plotly, Bokeh, Seaborn).

All plotting APIs return Matplotlib handles that can be further customized.
This allows users to create richer visualizations by adding overlays or modifying existing artists.
The full list of customizable parameters is available in the class docstrings.
Use the search functionality in the top-right corner to find specific functions.

Click on the links below to show some examples of the plotting functions available in pythermalcomfort.
The full list of functions is available in the "Plotting Functions Reference" section at the end of this page.

.. toctree::

    plots/matplotlib/threshold_plot.ipynb
    plots/matplotlib/summary_plot.ipynb

Examples Clear and Effective Visualizations
-------------------------------------------

In addition to the plotting API notebooks, this section includes practical plotting
recipes that combine `pythermalcomfort` with standard Matplotlib workflows for
time series, distributions, and overlays.

.. toctree::

    plots/matplotlib/example_plots.ipynb

Threshold Plot
--------------

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlot
    :members: __init__, set_x_axis, set_y_axis, set_params, set_regions, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlotResult
    :members:

Summary Plot
------------

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlot
    :members: __init__, set_regions, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlotResult
    :members:
