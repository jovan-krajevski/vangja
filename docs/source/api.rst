API Reference
=============

This section provides detailed API documentation for all modules in vangja.

.. contents:: Table of Contents
   :local:
   :depth: 2

Time Series Models
------------------

The time series module provides the base classes for building and composing
forecasting models. Models are built by combining components using arithmetic
operators.

Base Class
~~~~~~~~~~

.. autoclass:: vangja.time_series.TimeSeriesModel
   :members:
   :undoc-members:
   :show-inheritance:

Combined Models
~~~~~~~~~~~~~~~

These classes are created automatically when using operators to combine components.

.. autoclass:: vangja.time_series.CombinedTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vangja.time_series.AdditiveTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vangja.time_series.MultiplicativeTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: vangja.time_series.SimpleMultiplicativeTimeSeries
   :members:
   :undoc-members:
   :show-inheritance:

Components
----------

Components are the building blocks for time series models. They can be combined
using ``+`` (additive), ``*`` (simple multiplicative), or ``**`` (Prophet-style
multiplicative).

LinearTrend
~~~~~~~~~~~

.. autoclass:: vangja.components.LinearTrend
   :members:
   :undoc-members:
   :show-inheritance:

FlatTrend
~~~~~~~~~~~

.. autoclass:: vangja.components.LinearTrend
   :members:
   :undoc-members:
   :show-inheritance:

FourierSeasonality
~~~~~~~~~~~~~~~~~~

.. autoclass:: vangja.components.FourierSeasonality
   :members:
   :undoc-members:
   :show-inheritance:

NormalConstant
~~~~~~~~~~~~~~

.. autoclass:: vangja.components.NormalConstant
   :members:
   :undoc-members:
   :show-inheritance:

BetaConstant
~~~~~~~~~~~~

.. autoclass:: vangja.components.BetaConstant
   :members:
   :undoc-members:
   :show-inheritance:

UniformConstant
~~~~~~~~~~~~~~~

.. autoclass:: vangja.components.UniformConstant
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

Utility functions for data processing and model evaluation.

get_group_definition
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.get_group_definition

filter_predictions_by_series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.filter_predictions_by_series

metrics
~~~~~~~

.. autofunction:: vangja.utils.metrics

remove_random_gaps
~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.remove_random_gaps

compare_models
~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.compare_models

prior_sensitivity_analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.prior_sensitivity_analysis

plot_prior_posterior
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.plot_prior_posterior

plot_posterior_predictive
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.plot_posterior_predictive

plot_prior_predictive
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: vangja.utils.plot_prior_predictive


Datasets
--------

Functions for loading example datasets and generating synthetic data.

Real-World Datasets
~~~~~~~~~~~~~~~~~~~

load_air_passengers
^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.load_air_passengers

load_peyton_manning
^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.load_peyton_manning

load_citi_bike_sales
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.load_citi_bike_sales

load_nyc_temperature
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.load_nyc_temperature

Synthetic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

generate_multi_store_data
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.generate_multi_store_data

generate_hierarchical_products
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: vangja.datasets.generate_hierarchical_products

Types
-----

Type definitions used throughout vangja for type hints and documentation.

.. automodule:: vangja.types
   :members:
   :undoc-members:
   :show-inheritance:
