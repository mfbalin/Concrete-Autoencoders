# Concrete-Autoencoders

To install, use `pip install concrete-autoencoder`

To see how to use concrete autoencoders, you can take a look at this colab notebook:
https://colab.research.google.com/drive/11NMLrmToq4bo6WQ_4WX5G4uIjBHyrzXd

An implementation of the ideas in https://arxiv.org/abs/1901.09346.

There are 5 files.

"concrete_estimator.py" contains an implementation of Concrete Autoencoders in Tensorflow using the Estimators API.
"decoder.py" contains a decoder implementation to test selected indices for the GEO dataset.
"generate_comparison_figures.py" contains implementations of algorithms tested and datasets that were used.
"GEO_experiment.py" contains experiments for the GEO dataset.
"Concrete_Variable_Autoencoder_approach.ipynb" contains a jupyter notebook that was used to interactively generate the remaining figures and plots.

"scikit-feature-1.0.0" folder contains implementations of MCFS, UDFS and Lap Score. It was taken from "http://featureselection.asu.edu/index.php"
