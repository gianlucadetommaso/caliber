from typing import Optional

from caliber.multiclass_classification.minimizing.linear_scaling.calibration.cross_entropy_linear_scaling import (
    CrossEntropyLinearScalingMulticlassClassificationModel,
)


class DirichletMulticlassClassificationModel(
    CrossEntropyLinearScalingMulticlassClassificationModel
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
    ):
        super().__init__(
            minimize_options=minimize_options,
            has_intercept=True,
            has_shared_slope=False,
            has_cross_slopes=True,
            has_shared_intercept=False,
        )
