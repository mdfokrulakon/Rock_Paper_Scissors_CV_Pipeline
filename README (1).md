# Rock-Paper-Scissors Image Processing and Machine Learning

**A Reproducible Computer Vision Pipeline from Image Processing to Deep Learning**

A single, educational Jupyter notebook that builds a Rock-Paper-Scissors image classifier step by step, starting from basic image processing and progressing through traditional machine learning, neural networks, transfer learning, explainability, tuning, cross-validation, error analysis and a final locked evaluation.

## Dataset

[Kaggle: `drgfreeman/rockpaperscissors`](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors) — RGB hand images (about 300 × 200) in three classes. The notebook downloads it with KaggleHub; the dataset is **not** stored in this repository.

```python
import kagglehub
dataset_path = kagglehub.dataset_download("drgfreeman/rockpaperscissors")
```

Class mapping used everywhere: `paper = 0`, `rock = 1`, `scissors = 2`.

## What the Notebook Covers

`notebooks/rock_paper_scissors_complete.ipynb` contains 20 lessons in order:

| Part | Topic |
|---|---|
| 0 | Project setup (writes the reusable `src/` modules) |
| 1 | Data loading, EDA, data-quality checks, permanent split |
| 2 | Image processing (resize, crop, rotate, flip, filters, thresholds, edges, morphology) |
| 3 | HOG features + Logistic Regression / SVM / Random Forest |
| 4 | Artificial Neural Network |
| 5 | Baseline CNN |
| 6 | CNN with augmentation, BatchNorm and callbacks |
| 7 | Transfer learning with MobileNetV2 |
| 8 | CNN + HOG hybrid model |
| 9 | Model explainability and Grad-CAM |
| 10 | Reproducible pipeline and reusable modules |
| 11 | Ablation study |
| 12 | Manual hyperparameter tuning |
| 13 | KerasTuner |
| 14 | Augmentation experiment |
| 15 | Evaluation, bootstrap confidence intervals, seed variability |
| 16 | Stratified K-fold cross-validation |
| 17 | Error analysis |
| 18 | Model comparison |
| 19 | Final locked experiment (single test evaluation) |
| 20 | Final results and report |

## Quick Start

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cd notebooks
jupyter notebook rock_paper_scissors_complete.ipynb
```

Then choose **Kernel → Restart & Run All** (or run the cells from top to bottom).

> **Important:** open and run the notebook from inside the `notebooks/` folder. It uses relative paths (`../src/`, `../results/`).

## What Running the Notebook Creates

```text
<repo>/
├── notebooks/
│   └── rock_paper_scissors_complete.ipynb
├── src/                  # written by Part 0 of the notebook
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── models.py
│   └── evaluation.py
├── results/
│   ├── figures/
│   ├── metrics/          # dataset_split.csv and all metric files
│   ├── features/
│   └── models/
├── requirements.txt
├── README.md
└── .gitignore
```

## Methodological Rules

1. **Permanent split:** one stratified 70 / 15 / 15 train / validation / test split (`RANDOM_STATE = 42`) saved to `results/metrics/dataset_split.csv` and reused by every part. Corrupted files and exact (MD5) duplicates are removed first.
2. **No data leakage:** preprocessing objects such as `StandardScaler` are fitted on training data only.
3. **Augmentation** is applied to training data only.
4. **Test set:** not used for architecture selection, tuning or model selection. Models are compared on the validation set, and the final model is evaluated on the test set in Part 19. (Part 14 also computes one test row for its selected augmentation model; it is not used for any decision.)
5. **No fabricated results:** this repository contains no pre-computed scores. Every metric is produced when the notebook is executed.

## Requirements and Runtime Notes

* Python 3.12 (the version the pinned packages were tested with) and the packages in `requirements.txt`.
* Kaggle access may require credentials; see the [KaggleHub documentation](https://github.com/Kaggle/kagglehub).
* Part 7 downloads ImageNet weights for MobileNetV2 on first use, so an internet connection is needed.
* The full run trains many models. A GPU is recommended; on a CPU it can take a long time.
* All parts run in one kernel, so memory use can be high. Each part starts with a cell that frees the previous part's variables. If the kernel is killed, restart it and run from the part that failed: earlier outputs are saved in `results/`, and later parts read them.
* Several parts read files written by earlier ones (for example saved models, `selected_config.json`, result CSVs), so run the parts in order.

## Limitations

* Validation scores are used for selection and are therefore optimistically biased; the test estimate from Part 19 is the unbiased one, and it carries sampling uncertainty because the test set is small.
* Most configurations are trained once, so small differences between models may fall within seed variability (quantified in Part 15).
* Results come from a single dataset and may not generalise to other cameras, backgrounds or lighting.
* Grad-CAM is a qualitative diagnostic and does not prove causality.

## Acknowledgements

Dataset: `drgfreeman/rockpaperscissors` on Kaggle, accessed through KaggleHub. Please see its Kaggle page for the author and license terms.

## License

Add a license of your choice (for example MIT) before publishing.
