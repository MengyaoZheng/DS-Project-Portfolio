# DS Project Portfolio

## Mengyao Zheng 
[Linkedin @Mengyao Zheng](https://www.linkedin.com/in/mengyao-zheng/)


<h1 align="center">Causal Inference on the average treatment effect (ATE) of quitting smoking on weight gain</h1>

I have performed a causal analysis on a real-world healthcare dataset, known as the *NHANES I Epidemiologic Follow-up Study (NHEFS)* dataset. It is a government-initiated longitudinal study designed to investigate the relationships between clinical, nutritional, and behavioral factors. For more detail, please see [here](https://wwwn.cdc.gov/nchs/nhanes/nhefs/default.aspx/).

The task is to estimate the average treatment effect (ATE) of quitting smoking ($T$) on weight gain ($Y$). The NHEFS cohort includes 1,566 cigarette smokers between 25 - 74 years of age who completed two medical examinations at separate time points: a baseline visit and a follow-up visit approximately 10 years later. Individuals were identified as the treatment group if they reported smoking cessation before the follow-up visit. Otherwise, they were assigned to the control group. Finally, each individual’s weight gain, $Y$, is the difference in *kg* between their body weight at the follow-up visit and their body weight at the baseline visit. 

The notebook includes the following parts:
1. how a mechanism of how confounders, when unadjusted, can introduce bias into the ATE estimate. 
2. Implement propensity score re-weighting to estimate the ATE in Python.
3. Implement covariate adjustment strategies to estimate the conditional average treatment effect (CATE) as well as ATE in Python.
4. Assess how robust the ATE estimate is against potential unobserved confounders via sensitivity analysis. 

<h1 align="center">NLP: Context-Aware Legal Case Citation Prediction Using Deep Learning</h1>

Please see the video presentation here: [Presentation](https://www.youtube.com/watch?v=QfXUCw_XsT4)

We have craped 100K+ legal texts from Harvard Law School’s database using Python Scrapy, cleaned and tokenized. And we predicted in-text citation by both supervised and unsupervised learning methods: 1) building LSTM and CNN models with embedding layers using TensorFlow, 2) leveraging Legal-BERT model to obtain the embeddings and then used FAISS index to do similarity-based modelling. The Top model (LSTM) achieved a 200x accuracy boost as compared to the baseline random model.

<h1 align="center">AD_fbi: Automatic Differentiation Python Package</h1>

## Introduction

Our software package, *AD_fbi*, computes gradients using the technique of automatic differentiation. Automatic differentiation is important because it is able to solve the derivatives of complex functions at low cost while maintaining accuracy and stability. Its high practicality and precision make it applicable to a wide range of tasks, such as machine learning optimization, calculating financial loss or profit, and minimizing the cost of vaccine distribution. Automatic differentiation plays a crucial role in solving these problems. 

Prior to the technique of automatic differentiation, the two traditional computational techniques for solving differential equations are numerical differentiation and symbolic differentiation. The numerical differentiation method computes derivatives based on a finite numerical approximation which lacks stability and precision, and thus it is not ideal to be implemented in computer software. On the other hand, symbolic differentiation can become very computationally expensive as the function becomes more and more complex. 

The automatic differentiation technique transforms the entire computational process into a sequence of elementary arithmetic operations -- addition, subtraction, multiplication, division, and elementary functions. And then, it applies the chain rule on top of these elementary operations. In this way, it reduces the computational complexity that the model can reuse some of its parts computed in previous steps multiple times without re-calculating them. It keeps the same level of accuracy and precision as using symbolic differentiation. 

Our software package computes derivatives using the forward mode of auto differentiation. On top of that, it also implements different optimizers using leveraging auto differentiation, enabling users to find the minimum value of a function, location of the minimum value, and wall clock time to find these values.   

Here is the introduction of our implementation:


## Implementation
    
* __Main Classes__
  * <code>DualNumbers</code>: class for operations with a dual number.
  * <code>ForwardMode</code>: class for forward mode differentiation.
  * <code>Optimizers</code>: class for three different optimizers - Gradient Descent, Momentum,Adagradient. This object has no attributes, but each method within the
   class requires a <code>x</code> input, a function <code>func</code> input, and the number of iterations <code>num_iter</code> for the specific optimizing method. Additionally, each method
   has their own optional hyperparameters which the user can input if they choose not to use our standard default values.

* __Core Data Structures & Dual Numbers__
  * Our primary core data structure is a numpy array, which we use to store both the variable list and the function list. Then using the
   methods within the `ForwardMode` class, we compute the jacobian and function value. We store the corresponding values or arrays in a tuple. For a single input, the jacobian and function value are singular values, and for multi-dimensional vector input, the stored values are arrays.


* __Classes Method & Name Attributes__
  * <code>DualNumbers</code>: 
    * A `__init__` method to initialize a <code>DualNumbers</code> object with a real number value and a dual number value.
    * A `__repr__` method to return the object representation in string format.
    * Multiple methods to overload the elementary operations for a dual number. e.g. `__add__`, `__sub__`, `__mul__`, `__div__`, `__pow__`, `__radd__`, `__rsub__`, `__rmul__`, `__rdiv__`, `__rpow__`, etc.
    * Multiple methods to compare dual numbers. e.g. `__ne__`, `__eq__`, etc.
    * Multiple methods to transform a dual number. e.g. `sqrt`, `log`, `sin`, `cos`, `exp`, `tan`, etc.

  * <code>ForwardMode</code>:
    * A `__init__` method  to initialize a `ForwardMode` Object with an input value `x`, a function `func`, and a seed vector `seed`.
    * A `get_fx_value` method to run the forward mode process and return a function value at the evaluated point `x`.
    * A `calculate_dual_number` method to run the forward mode process and return the evaluated value and the derivative of the input function at the evaluated point `x`.
    * A `get_derivative` method to run the forward mode process and return a value of directional derivative corresponding to a specific seed vector.

  * <code>Optimizer</code>:
    * A `gradient_descent` method implements the Gradient Descent optimizer that takes in the inputs of `x`, `func` and `num_iter`. Users can change the learning rate `alpha` by specifying custom learning rate.
    * A `momentum` method implements the momentum optimizer that takes in the inputs of `x`, `func` and `num_iter`. Users can change the learning rate and momentum coefficient `alpha` and `beta` by specifying their custom values.
    * A `ADAGRAD` method implements the Adaptive gradient optimizer that takes in the inputs of `x`, `func`, and `num_iter`. Users can change the learning rate and a small denominator number `alpha` and `epsilon` by specifying their custom values.

  
* __Dealing With Operator Overloading and Elementary Functions__

   * For the overloading operator template (like `__add__` for our special dual number class object), we implemented multiple methods such as `__radd__`, `__rsub__`, `__rmul__`, etc. to handle both cases of dual number + integer, and integer + dual number. The `__r*__` methods are necessary to handle overloading. 

   *  As listed above, within the <code>DualNumbers</code> class we’ve overloaded the simple arithmetic functions (addition, subtraction, multiplication,
   division, negation, power, equal, and not equal) to calculate both the value and the dual number. We’ve also defined our own
   elementary functions, such as `sin` and `sqrt` etc. to compute the value and the derivative. This module
   generalizes each of the functions in order for the <code>ForwardMode</code> class to handle both scalar and vector inputs. Each method has also implemented raise error attribute to deal with all possible types of invalid inputs. The output is a tuple of the function value and the derivative, which
   is used in the <code>ForwardMode</code> class.
   
* __Dealing With Operator Overloading on Reverse Mode__

   * We currently are not interested in implementing a reverse mode on our package.

* __Dealing With MultiDimensional Input and Output Function__

   * Use `try` and `except` to handle multi-dimensional and single-dimensional case separetly. For the multi-dimensional case, we design a helper function to loop through all of the functions' inputs and reassign the value/derivative as a vector. 
   * We treat functions as a list (so high dimensional functions will be a list of functions)
   * The `grad()` function (or jacobian function) is generic to both single-dimensional and multi-dimensional(as they are either a list of 1 or a list of mulitple functions).
   
* __External Dependencies__
   * We use the numpy library to create our data structure for the computational graph and perform
   computations outside of those we created in our dual_number class.

## License

Our *AD_fbi* package is licensed under the GNU General Public License v3.0. This free software license allows users to do just about anything they want with our project, except distribute closed source versions. This means that any improved versions of our package that individuals seek to release must also be free software. We find it essential to allow users to help each other share their bug fixes and improvements with other users. Our hope is that users of this package continually find ways to improve it and share these improvements within the broader scientific community that uses automatic differentation.

