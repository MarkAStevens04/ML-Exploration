# ML-Exploration
This repo is designed to allow for easy research on different ML models,
the ways they perform, and their potential pitfalls/downfalls.
It's also meant to be a playground for testing experimental models.

# General Purpose
Investigate many types of data generation, and many types of models.

The goal is to have a vast array of different models & data types.
Having them in one place allows for direct and easy **comparison**. See what
types of data a single model performs best on, see what types of models
are best suited for different types of features, etc.

The goal is to have a deep understanding of the trends we see in data
and how the models typically behave on that type of data. That way, when
we don't know the underlying trend, we're able to make better judgments
about our results.

We should also be able to see **when** models typically fail, and how we can
**combine ideas** from multiple models to generate even more effective models.

# Goals
- Many different **models**
- Many different **data** patterns
  - Clear understanding of the underlying behavior of the data
- Understand _why_ models behave the way they do in certain conditions.
  - Example: if a regression model is given a sine wave, show how close its predictions are to the true value.
- Hot-swappable
  - Make quick and easy comparisons between many different types of models.
  - Performance may not be optimal, but the trends drawn should be powerful.
  - The training may take a long time, but the programming should be quick.
- Understanding over performance
  - Multi-threading and other techniques can speed up certain processes. But if this impacts our ability to understand what's going on, we should find another way. 



# Ideas

### Classification - Shapes of data
Imagine a 2D plane, each axis is one feature. The color of the data
will represent the class it belongs to (imagine red / blue).
Now, see if models perform better, worse, or similar when the red region
is an area versus a line. And see if it can handle "loops". It's technically
not an invertible function!

See if it works best with areas/lines, volumes/hyper-planes, etc.


### Classification - Number of classes
See if the number of classes affects any of the models.
Are models okay if they have 3/10/100/1000+ classes?

### Regression - Number of classes
Do models behave well when there are 3/10/100/1000+ classes?


### Fitting - "unstuck"
Get your model unstuck from local minima. I remember watching a video where a programmer seemingly
combed through each parameter, resetting it to the mean (?) and then continuing to fit the model.
The model was able to find the global minimum!

Look into ways of getting models out of local minima.

### Model Mixture - Layers of models
Okay, imagine you train a model to balance a metal rod or something.
But you trained the model in Unreal engine, not in real life! So you need an
**adapter model**. One that takes the messy, real-world input, then converts
it into clean data for our pre-trained model.

Essentially
- Train a model on ideal inputs
- Train a second model to take some non-ideal inputs, then convert them into semi-ideal inputs
- Then, we could even recurse back, and continue training our first "optimistic" model. You'd have to explore ways of getting models "unstuck" from local minima.

We could also try adding an extra feature & hidden node to every layer in our 
pre-trained model. This way, any information we *gained* from the messy data
is retained and iterated on. This adding a neuron shouldn't be too bad, it's
like the combing to find a global optimum.


### Data - Randomness and sample size
Does the number of datapoints increase accuracy if there's inherent randomness in the dataset?

Set your data to have some underlying randomness to it. Like pretend the maximum
possible accuracy iss 90% because of the randomness of your data.
See if increasing the number of datapoints causes the network to perform better?

Or, add some randomness to like 100 different features, and see if increasing the number
of datapoints really helps.

Compare having randomness versus no randomness, and lots of datapoints vs few datapoints.


### Data - Missing segments
Is it okay if some segments of data are removed?

simulate not having perfect data-collection, and occasionally remove datapoints.
Try removing them randomly, or removing them at specific points (segments)
with a specific probability distribution. See how this affects accuracy!

### Data - Outliers
See what happens when you have a few outliers, or a few pieces of key data removed.
Simulate imperfect data collection.

How does it affect when the model overfits? W


### Accuracy - find poor performing clusters
Use a clustering algorithm like k-means to find clusters of data the model
typically performs poorly on. 

My initial guess is to try fitting a model, then fitting a k-means algo, then assessing
the likelihood of a correct classification in each category. See if any category
performs especially poorly.

Or, take the inputs the model performs poorly on. 

Use some naive algorithm (Mixture, GDA, Bayesian) to draw trends on
data we're performing poorly on.


### General structure - Hyperparam tuning
Maybe have an additional (optional) function to models that need hyper-parameter
tuning. The function would return models with all combinations of the provided hyper-parameters.

When training multiple types of models with multiple hyper-parameter combinations,
maybe have a parent class function that handles this? Like, instead of
manually coding iteration through 20 models every time you want to try a new
combination of models and data, just have two parent methods that takes in some models
as a parameter. One method does a partial fit, and the mother does a complete training.

### General structure - Data visualization
Have a data-visualization suite. Maybe have a data-visualization suite
for each type of data we generate? That way, it's super easy to see how models
perform with a specific type of data.


### General structure

We have "Harness" parent class, and other explorations inherit this.
The Harness class provides a general outline:

- Collect & Format data (data path is a parameter)
- Train models (models, hyper-params, and training / validation data are parameters)
- Score & Visualize (train models & plot their performance)


Here's some methods a basic harness would implement:
- Data:
  - Collect data, train/test split, and shuffle.
- Training:
  - Train an MLP & Lin-reg model from the training data
  - Store the validation and training accuracy at specified intervals
- Prediction:
  - Give both models the validation data, and return the predictions.
  - Maybe the prediction is done on a per-model basis? 
- Visualization
  - Turn the prediction and the training data into a graph.
  - Show the difference in prediction and the training data.
  - Shows the types of data the models typically gets wrong