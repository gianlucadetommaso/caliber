from typing import Optional

from caliber.binary_classification.minimizing.linear_scaling.calibration.cross_entropy_linear_scaling import (
    CrossEntropyLinearScalingBinaryClassificationModel,
)


class BetaBinaryClassificationModel(CrossEntropyLinearScalingBinaryClassificationModel):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        num_features: int = 0,
    ):
        super().__init__(
            minimize_options=minimize_options,
            has_intercept=True,
            has_bivariate_slope=True,
            num_features=num_features,
        )


class DiagBetaBinaryClassificationModel(
    CrossEntropyLinearScalingBinaryClassificationModel
):
    def __init__(
        self,
        minimize_options: Optional[dict] = None,
        num_features: int = 0,
    ):
        super().__init__(
            minimize_options=minimize_options,
            has_intercept=True,
            has_bivariate_slope=False,
            num_features=num_features,
        )
