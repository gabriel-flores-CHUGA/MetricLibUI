from metriclib.metric import StreamMetric, TabularMetric, MetricResult
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
