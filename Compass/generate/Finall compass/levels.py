from dataclasses import dataclass
from typing_extensions import assert_type


@dataclass
class InnerLevel:
    """Inner CLEVA-Compass level with method attributes."""

    Proftability: int
    Risk_Control: int
    University: int
    Diversity: int
    Reliability: int
    Explainability: int
    

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        cleva_template.tex file.
        """
        for item in [
            self.Proftability,
            self.Risk_Control,
            self.University,
            self.Diversity,
            self.Reliability,
            self.Explainability,
        ]:
            yield item


@dataclass
class OuterLevel:
    """Outer CLEVA-Compass level with measurement attributes."""

    alpha_decay: bool
    profit: bool
    extreme_market: bool
    risk_adjusted: bool
    risk: bool
    time_scale: bool
    assert_type: bool
    country: bool
    rolling_window: bool
    correlation: bool
    entropy: bool
    t_SNE: bool
    rank_order: bool
    variability: bool
    profile: bool
    equity_curve: bool
    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        cleva_template.tex file.
        """
        for item in [
            self.alpha_decay,
            self.profit,
            self.extreme_market,
            self.risk_adjusted,
            self.risk,
            self.time_scale,
            self.assert_type,
            self.country,
            self.rolling_window,
            self.correlation,
            self.entropy,
            self.t_SNE,
            self.rank_order,
            self.variability,
            self.profile,
            self.equity_curve,
        ]:
            yield item


@dataclass
class CompassEntry:
    """Compass entry containing color, label, and attributes."""

    color: str  # Color, can be one of [magenta, green, blue, orange, cyan, brown]
    label: str  # Legend label
    inner_level: InnerLevel
    outer_level: OuterLevel
