# AdaBoost from scratch (2D)

This repository provides a minimal, from‑scratch implementation of AdaBoost using 2D decision stumps. It includes:

- **`adaboost.py`** – the core `AdaBoost` and `DecisionStump` classes  
- **`adaboost_demo.ipynb`** – an interactive notebook showing how to train, visualize each round, and (optionally) build an animated GIF  

---

## 🚀 Features

- Implements AdaBoost with simple decision stumps in 2D  
- Dynamic plotting of the ensemble decision boundary after each round  
- Optional GIF animation (with Pillow)  
- **Minimal dependencies** and under 150 lines of code

---

## 📁 Repository Structure

```text
.
├── images/               # Saved plots for each iteration (will be created during the first run)
├── resources/            # GIF examples and pdf on the theory
├── utils.py              # GIF generation & data generation
├── adaboost.py           # AdaBoost & DecisionStump implementation
├── adaboost_demo.ipynb   # Example usage in Jupyter
├── adaboost.gif          # Output animation (will be created during the first run)
└── README.md

```

---

## 📦 Requirements

- Numpy
- Matplotlib
- Pillow  (if you set `create_gif=True`)

---

## 🧪 How to Run

You can test the algorithm and generate visualizations with the included notebook `adaboost_demo.ipynb`. This repo has been designed to be simple to understand and modify.

The `make_moons` function in `utils.py` is based on the `make_moons` function from `sklearn.datasets` and has been reimplemented to avoid any dependencies other than `numpy`. It generates a dataset of two interleaving half circles (common test case for binary classifiers).

## Documentation:

**_class_ `AdaBoost`:**
- **Parameters:**
    - `n_classifiers` (`int`): number of weak learners (decision stumps) to train

**Other parameters:**
- `plotting` : whether to display the plot interactively
- `create_gif` : whether to create a GIF animation of the results (needs pillow package)


## 📈 Output

The `AdaBoost` class saves visualizations of each iteration in the `images/` folder. When it is done and if `create_gif` is set to `True`, it generates an animated GIF (`adaboost.gif`) illustrating how the decision boundary evolves over time.

![AdaBoost animation](https://github.com/paulbouuu/AdaBoost-from-scratch/raw/main/resources/adaboost_example.gif)

### License
This project is free to use and modify under the MIT License. See the [LICENSE](LICENSE) file for details.