# Sonador 3D Python Library
Sonador 3D provides libraries and integrations to transform medical data into 3D models for use in surgical navigation, 3D printing, and other advancecd research. It builds on top of the [Sonador IO Python library](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-client) to provide tools for converting [pixel/payload](https://code.oak-tree.tech/oak-tree/medical-imaging/3d/sonador3d-imaging#) and spatial data to the [Simplified ITK toolkit (SimpleTK)](https://simpleitk.org/) and [PyVista](https://pyvista.org/).

Dependencies:

* [Sonador IO Python library](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-client) provides [client models](https://code.oak-tree.tech/django-apps/guru/-/wikis/dev.client-tools) for retrieving medical data from [Sonador/Orthanc](https://code.oak-tree.tech/oak-tree/medical-imaging/orthanc-sonador).
* [Simplified Insight Toolkit (SimpleITK)](https://simpleitk.org/): open-source multi-dimensional image analysis tools for Python, R, Java, C#, Lua, Ruby, and C++. Provides a simplified and unified interface to the [Insight Toolkit (ITK)](https://itk.org/).
* [PyVista](https://pyvista.org/): framework for working 3D data built on top of the [Visualization Toolkit (VTK)](https://vtk.org/). VTK provides toolks for representing, manipulating, and displaying scientific data and combines systems for 3D rendering, suite of widgets for 3D interaction, and plotting capabilities.

This repository unifies two related libraries:

* [Sonador 3D Imaging](https://code.oak-tree.tech/oak-tree/medical-imaging/3d/sonador3d-imaging): provides the base classes and core interfaces needed for working with "imaging volumes" (3D representations of pixel data) and interfacing with SimpleITK.
* Sonador 3D Spatial: provides the base classes and core interfaces with working with "spatial data" (meshes, point clouds, and other VTK/PyVista objects).

The two lower-level libraries are packaged separately to allow for them to deployed without requiring all of the Simple ITK or VTK/PyVista dependencies in applications which do not not use features from all three frameworks. **It is suggested that application developers install this library.** `sonador3d_imaging` and `sonador3d_spatial` are direct dependencies and will also be installed. Features from both (along with integration helpers and classes) can be accessed from the `sonador3d.imaging` and `sonador3d.spatial` modules.


## Installation
Pre-built binaries of Sonador 3D are available from the [Sonador Python package](https://code.oak-tree.tech/oak-tree/medical-imaging/imaging-development-env/-/wikis/deployment.package-repository) repostiory and can be installed via `pip`:

```bash
pip install sonador sonador3d --extra-index-url https://code.oak-tree.tech/api/v4/projects/335/packages/pypi/simple
```

To install from source, clone this repository and install the dependencies.


## Usage
Examples of how to utilize the features in Sonador 3D can be found in the [Sonador Examples Repository](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples). _More advanced capabilities visualization, working with meshes (and point cloud data), and how to convert between pixel/spatial representations of data are demonstrated in the [`technical-reference` subfolder](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/tree/master/technical-reference?ref_type=heads)._

* Visualization
  - [Basic Visualization in JupyterLab](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/blob/master/technical-reference/sonador-jupyter.visualization-view3d.ipynb?ref_type=heads)
  - [Part 1](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/blob/master/technical-reference/sonador-jupyter.visualization-view3d.ipynb?ref_type=heads): preparation and staging of integrated datasets. Uses data from the [Denver University Natural Knee](https://digitalcommons.du.edu/natural_knee_data/) project.
  - [Part 2](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/blob/master/technical-reference/sonador3d.natural-knee02.visualization.ipynb?ref_type=heads): visualization and interaction within [JupyterLab](https://jupyter.org/) using the [Sonador Jupyter](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-jupyter) widgets.
* Meshes
  - [Creating Encapsulated STL Files](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/blob/master/technical-reference/sonador3d.stl-encoding.ipynb?ref_type=heads).
  - [Creating Mesh Geometries from Segmentations](https://code.oak-tree.tech/oak-tree/medical-imaging/sonador-examples/-/blob/master/technical-reference/sonador3d.segmentation2mesh.ipynb?ref_type=heads). Describes how Sonador (via PyVista and VTK) can be used to create triangulated meshes and other geometry files from mask based segmentations. Also includes example code of how to run advanced operations such as downsampling and re-surfacing.

  