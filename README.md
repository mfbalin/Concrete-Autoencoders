# Concrete-Autoencoders

To install, use `pip install concrete-autoencoder`

To see how to use concrete autoencoders, you can take a look at this colab notebook:
https://colab.research.google.com/drive/11NMLrmToq4bo6WQ_4WX5G4uIjBHyrzXd

An implementation of the ideas in https://arxiv.org/abs/1901.09346.

Description:

Concrete Autoencoders is a kind of an autoencoder, which does feature selection when you train it.

Documentation:


class ConcreteAutoencoderFeatureSelector:

  Constructor takes a number of parameters to initalize the class. They are:
    K: the number of features one wants to select
    output_function: the decoder function
    num_epochs: number of epochs to start training concrete autoencoders
    batch_size: the batch size during training
    learning_rate: learning rate of the adam optimizer used during training
    start_temp: the starting temperature of the concrete select layer
    min_temp: the ending temperature of the concrete select layer
    tryout_limit: number of times to double the number of epochs and try again in case it doesn't converge
    
  fit(X, Y = None): trains the concrete autoencoder
    X: the data for which you want to do feature selection
    Y: labels, in case labels are given, it will do supervised feature selection, if not, then unsupervised feature selection
  
  transform(X): filters X's features after fit has been called
    X: the data to be filtered
    
  fit_transform(X): calls fit and transform in a sequence
    X: the data to do feature selection on and filter
    
  get_support(indices = False): if indices is True, returns indices of the features selected, if not, returns a mask
    indices: boolean flag to determine whether to return a list of indices or a boolean mask
  
  get_params(): returns the underlying keras model for the concrete autoencoder
    
