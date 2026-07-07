"""model-distillation — distill Claude's topic-classification behavior into a local TF-IDF+LogReg
student, and measure how much quality survives, what the noisy teacher costs, and when distilling
pays off."""
from .benchmark import format_report, run

__all__ = ["run", "format_report"]
