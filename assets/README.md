# Figures

Paper figures, rasterized from the vector figures of the camera-ready
manuscript for display on GitHub and Hugging Face:

| File | Paper figure |
| --- | --- |
| `overview.png` | Figure 1: token-level loss trajectories and the learning-time spectrum versus the loss derivative along T, D, and M |
| `token_learning_events.png` | Figure 2: representative token trajectories across independent runs, with sigmoid fits and learning pulses |
| `t_axis.png` | Figure 3: training-step axis decomposition |
| `d_axis.png` | Figure 4: data-scale axis decomposition |
| `m_axis.png` | Figure 5: model-scale axis decomposition |
| `reshape.png` | Figure 6: static reshaping with measured learning times |
| `predictive_reshaping.png` | Figure 7: dynamic predictive reshaping |

`readme/` holds the images of the README and the dataset card, drawn by
`tools/make_readme_assets.py` in the style of the project page: `banner.png`
(the curves are the same 46 tokens as in the social preview below),
`btn-*.svg` (link buttons, 40 px tall, with the labels as outlines), and `mechanism.gif` (a preview of
the page's mechanism toy, with its seed and default sliders).

`social_preview.png` (1280 × 640) is the repository's social preview card and the
project page's link preview. Its colored curves are normalized sigmoid fits of
46 tokens from the paper-setting reproduction of Figure 3 (`step_axis_1m`),
chosen among the 950 tokens learned between steps 25k and 60k in a sharp step
(10–90% width under 3,500 steps, Δ > 0.3, R² > 0.6); the black curve averages
all 950.
