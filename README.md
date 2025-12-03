# Financial Machine Learning Research

This repository supports the research conducted for my Master's dissertation on Financial Machine Learning. The project is structured to keep experiments, code, and analysis well organized and reproducible as the work develops.

## Project Structure

```
financial-ml-research/
│
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ .gitignore
│
├─ data/
│   ├─ raw/
│   ├─ interim/
│   └─ processed/
│
├─ notebooks/
│
├─ src/
│   └─ financial_ml/
│       (contents of this package will be defined gradually during the project)
│
├─ experiments/
│   ├─ configs/
│   ├─ results/
│   └─ runners/
│
└─ docs/
```

### Directory descriptions

- **data/** contains local datasets used throughout the research. This directory is excluded from version control and is split into raw, intermediate, and processed forms of the data.
- **notebooks/** holds Jupyter notebooks for exploration, early experimentation, and visualization.
- **src/financial_ml/** will contain reusable modules developed over the course of the dissertation. Its internal structure is intentionally left open until the project takes shape.
- **experiments/** groups all experiment-related material: configuration files, generated results, and scripts that execute experiments. The aim is to ensure that each experiment can be rerun and documented clearly.
- **docs/** stores supporting material for the dissertation, such as figures, notes, or other written components.

## Experiment workflow

Each experiment is defined through a configuration file in `experiments/configs/`. A runner script in `experiments/runners/` loads the configuration and executes the experiment according to the parameters specified there. Results are saved automatically into a dedicated folder under `experiments/results/`.

This structure cleanly separates code, configurations, and outputs, making it easier to document, repeat, and compare experiments throughout the dissertation. Running an experiment only requires pointing the runner to the corresponding configuration file, without modifying the core code.

## Notes

The project is research-oriented and not intended as a production system. The layout of the `financial_ml` package will evolve naturally as the dissertation progresses.s