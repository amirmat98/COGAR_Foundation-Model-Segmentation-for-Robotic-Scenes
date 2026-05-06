"""Analysis helpers for segmentation evaluation outputs."""

from cogar_seg.analysis.comparison import (
    PromptComparisonRun,
    compare_prompt_results,
)
from cogar_seg.analysis.results import (
    PromptAnalysisRun,
    analyze_prompt_results,
)

__all__ = [
    "PromptAnalysisRun",
    "PromptComparisonRun",
    "analyze_prompt_results",
    "compare_prompt_results",
]
