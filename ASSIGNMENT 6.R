#Install and Load the necessary libraries
install.packages('keras')
library(keras)
library(tensorflow)

# Load and preprocess the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Normalize the data
train_images <- train_images / 255
test_images <- test_images / 255

# Reshape the data
train_images <- array_reshape(train_images, c(nrow(train_images), 28, 28, 1))
test_images <- array_reshape(test_images, c(nrow(test_images), 28, 28, 1))

# Class names
class_names <- c('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                input_shape = c(28, 28, 1)) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'sparse_categorical_crossentropy',
  metrics = c('accuracy')
)

# Train the model with early stopping
early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3, restore_best_weights = TRUE)
model %>% fit(train_images, train_labels, epochs = 50, validation_split = 0.2, batch_size = 64, callbacks = list(early_stopping))

# Evaluate the model
score <- model %>% evaluate(test_images, test_labels)
cat('Test accuracy:', score$accuracy, '\n')

# Predict on the first two test images
predictions <- model %>% predict(test_images[1:2, , , drop = FALSE])

# Function to plot the predictions
plot_predictions <- function(images, predictions, labels) {
  par(mfrow = c(1, 2))
  for (i in 1:2) {
    image(1:28, 1:28, t(apply(images[i, , , 1], 2, rev)), col = gray.colors(256), axes = FALSE)
    title(paste("Predicted:", class_names[which.max(predictions[i, ])], 
                "(True:", class_names[labels[i] + 1], ")"))
  }
}

# Plot predictions for the first two images
plot_predictions(test_images[1:2, , , drop = FALSE], predictions, test_labels[1:2])
