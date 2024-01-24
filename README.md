# California Housing Regression with Custom Huber Loss

Explore regression on the California Housing dataset with a custom Huber loss function implemented using TensorFlow and Keras. This project showcases the creation of a neural network for regression, emphasizing the flexibility of custom loss functions in handling outliers.

## Features:

- **California Housing Dataset:**
  - Utilizes the California Housing dataset for regression tasks, featuring various housing-related features.

- **Custom Huber Loss Function:**
  - Demonstrates the implementation of a Huber loss function, combining Mean Squared Error (MSE) and Mean Absolute Error (MAE) for robust regression.

- **Neural Network Model:**
  - Creates a neural network model with SELU activation and lecun_normal initialization for regression.

- **Nadam Optimizer:**
  - Optimizes the model using the Nadam optimizer during training.

- **Model Training and Evaluation:**
  - Trains the model on the California Housing dataset and evaluates its performance on a test set.

## Usage:

1. **Clone the Repository:**
   - Clone this repository to your local machine using `git clone`.

2. **Explore the Regression Implementation:**
   - Open the provided Jupyter Notebook in a compatible environment to dive into the regression model implementation.

3. **Run the Code:**
   - Execute the code cells to observe the training progress and evaluate the regression model.

## Custom Huber Loss Function:

The custom Huber loss function is defined as follows:

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
