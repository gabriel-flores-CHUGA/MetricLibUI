from metriclib.metric import StreamMetric, TabularMetric, MetricResult
from metriclib.metrics.measurement_process import DICESimilarityCoefficient, IntersectionOverUnion, HausdorffDistance, HausdorffDistance95
import numpy as np
from abc import ABC, abstractmethod


class CustomMetric:
    def __init__(self):
        pass


# class AgeMean(CustomMetric, TabularMetric):
#     """A custom metric that calculates the mean of a numeric field."""
# 
#     def __init__(self):
#         self.dimension = "variety_age"
# 
#     def compute(self, data, **kwargs):
#         return MetricResult(
#             value=data["age"].mean(),
#             cluster="Representativeness",
#             description="Mean age in the dataset",
#         )


class DICEMean(CustomMetric, DICESimilarityCoefficient):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "dice_coefficient"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        DICESimilarityCoefficient.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="DICE mean",
        )

class DICEMedian(CustomMetric, DICESimilarityCoefficient):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "dice_coefficient"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        DICESimilarityCoefficient.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="DICE median",
        )

class IntersectionOverUnionMean(CustomMetric, IntersectionOverUnion):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "intersection_over_union"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        IntersectionOverUnion.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="Intersection over Union mean",
        )

class IntersectionOverUnionMedian(CustomMetric, IntersectionOverUnion):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "intersection_over_union"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        IntersectionOverUnion.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="Intersection over Union median",
        )
        
class HausdorffDistanceMean(CustomMetric, HausdorffDistance):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="HD mean",
        )

class HausdorffDistanceMedian(CustomMetric, HausdorffDistance):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="HD median",
        )

class HausdorffDistance95Mean(CustomMetric, HausdorffDistance95):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance95"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance95.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.mean(data),
            cluster="Measurement Process",
            description="HD95 mean",
        )

class HausdorffDistance95Median(CustomMetric, HausdorffDistance95):
    """A custom metric that calculates the mean of a numeric field."""

    def __init__(self):
        self.dimension = "hausdorff_distance95"
        
    def aggregate(self, datapoint, reference=None, metric_config=None): 
        HausdorffDistance95.aggregate(datapoint, reference, metric_config)

    def compute(self, data, **kwargs):
        return MetricResult(
            value=np.median(data),
            cluster="Measurement Process",
            description="HD95 median",
        )


        
#
# Example of a custom stream metric that
# class ECGNoise(CustomMetric, StreamMetric):
#    """A custom metric that calculates the noise level in ECG signals."""
#
#    def __init__(self):
#        self.dimension = "signal_precision"
#
#    def aggregate(self, datapoint, reference=None, metric_config=None):
#        return np.std(np.array(datapoint[0]))
#
#    def compute(self, data, **kwargs):
#        return MetricResult(
#            value=np.array(data).mean(),
#            cluster=None,
#            description="Average noise level in ECG signals",
#        )


class CustomChart(ABC):
    """Base class for custom charts.

    Subclasses are auto-discovered via `CustomChart.registry` (populated by
    `__init_subclass__`, the same mechanism `TabularMetric`/`StreamMetric` use
    for `CustomMetric`) and rendered once per dataset by the backend, so no
    manual `report.add_chart(...)` call is needed.
    """

    registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        CustomChart.registry[cls.__name__] = cls

    @abstractmethod
    def render(self, data, **kwargs):
        """Build and return a plotly figure for the given dataset metadata.

        Parameters
        - data: pd.DataFrame
                The dataset's processed metadata (the same data passed to
                `TabularMetric.compute`).

        Returns
        - a plotly.graph_objects.Figure (or plotly.express figure)
        """
        raise NotImplementedError()


# Example of a custom chart that plots the distribution of patient age
# import plotly.express as px
#
# class AgeHistogram(CustomChart):
#    """A custom chart that shows the distribution of patient age."""
#
#    def __init__(self):
#        self.dimension = "variety_age"
#
#    def render(self, data, **kwargs):
#        return px.histogram(data, x="age", title="Age distribution")
