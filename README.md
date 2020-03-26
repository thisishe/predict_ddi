# Novel deep learning model for more accurate prediction of drug-drug interaction effects
We introduce a deep learning model for prediction of drug-drug interaction effects accurately.

This model is composed of three autoencoders and a deep feed-forward neural network.

The features used for training the model are structural similarity profile (SSP), target gene similarity profile (TSP) and gene ontology similarity profile (GSP).

For more details, please refer to [here](https://doi.org/10.1186/s12859-019-3013-0)


### Python library requirements

python=3.6.8

pytorch=1.4.0

jupyter=1.0.0

scikit-learn=0.22.2

pandas=1.0.3

openbabel=2.4.1


### Example
		cd src
		python run.py
		
		
* model_evaluation.ipynb : Calculation of accuracy, macro recall, macro precision, micro recall, micro precision of trained model.
* structural_similarity_example.ipynb : Example of computing structural similarity using openbabel library. (trained model is necessary for implementation)

#### If you need trained model, visit [here](https://bitbucket.org/thisishe/drug-drug_interaction/src/master/)
