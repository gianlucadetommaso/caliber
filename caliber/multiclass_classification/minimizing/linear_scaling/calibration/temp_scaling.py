from typing import Optional

from caliber.multiclass_classification.minimizing.linear_scaling.calibration.cross_entropy_linear_scaling import (
    CrossEntropyLinearScalingMulticlassClassificationModel,
)


class TemperatureScalingMulticlassClassificationModel(
    CrossEntropyLinearScalingMulticlassClassificationModel
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
    ):
        super().__init__(
            minimize_options,
            has_intercept=False,
            has_shared_intercept=False,
            has_shared_slope=True,
            has_cross_slopes=False,
        )
