MAVIS presentation figures - 29 October 2026

Six figures exported as 240 dpi PNG and editable vector SVG.
Use PNG for direct slide insertion; use SVG for scaling or editing labels.
All numbers come from 15 existing JSON files in results/raw/phase5_factorial.
No training was performed. No model, trainer, data-loader or configuration code was changed.

01_final_accuracy: M3 leads at 24.32%; M4 22.23%; M4noA 20.84%.
02_task_progression: seen-task and old-task performance across the sequence.
03_learning_and_forgetting: M4noA forgets less by the drop metric, but learns less strongly.
04_paired_comparisons: M4-M4noA hierarchical gain +1.39 pp; paired 95% CI [+0.07,+2.72].
05_task_heatmaps: final M4 gain +7.80 pp on newest task; +0.68 pp on older tasks on average.
06_predictor_comparison: M4 head 20.82%, hierarchical 22.23%; predictor choice matters.

Plots label SD versus confidence intervals explicitly. There are only three optimization
seeds under one class-order setup. Intervals are not adjusted for multiple comparisons.
M1/M2 use classifier-head prediction; M3/M4/M4noA use hierarchical prediction except
where the plotted predictor is explicitly labeled. Baseline comparisons change multiple
training and prediction components and historical data access.
M4noA keeps memory, replay, distillation, Online EWC and prototype-based prediction.
Its attention switch changes live fusion, replay fusion and reconstruction together.
The implementation revisits earlier training images for memory refresh.

plot_data.json contains per-seed plotted values, computed comparisons and source SHA256s.
Reproduce from the repository root:
  .venv\Scripts\python.exe experiments/plot_presentation.py
Requires numpy and matplotlib plus the existing experiments/analyze_factorial.py.
The saved source/config provenance is incomplete; these are analyses of saved runs,
not results from a new controlled retraining experiment.

The repository's docs/FACULTY_PRESENTATION_GUIDE.html includes the complete plot gallery
and presentation-ready interpretations.
