conda update -n base conda
conda create -n growth_hack python=3.7
conda activate growth_hack
conda install --yes -n growth_hack Altair=4.1.0 imbalanced-learn=0.8.1 NumPy=1.19.5 pandas=1.1.5 NLTK=3.2.5 SciPy=1.4.1 transformers=4.14.1 scikit-learn=1.0.1
pip install --quiet ndjson