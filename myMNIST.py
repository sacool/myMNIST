# One layer deep neural network. Achieves around 98% accuracy on a testing set.

# from preprocessed_mnist import load_dataset
# Keras only used for loading data
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime

plt.ion()

# Function to load and flatten the image data, one_hot_encode the label data


def load_dataset():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    X_train = X_train.reshape(
        X_train.shape[0], X_train.shape[1] * X_train.shape[2])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1] * X_val.shape[2])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
    with tf.Session() as sess:
        y_train = tf.one_hot(y_train, 10).eval()
        y_val = tf.one_hot(y_val, 10).eval()
        y_test = tf.one_hot(y_test, 10).eval()
    print('data loaded')
    return(X_train, y_train, X_val, y_val, X_test, y_test)


# Define the model:

def model_2layer_dropout(n_units_1=512, n_units_2=256, batch_size=200, num_steps=3300):
    # Set some parameters for how often losses and accuracies are displayed:
    # Measure losses 20 times throughout training
    record_every = (num_steps / 20) // 1
    # Measure and print accuracy 6 times throughout training:
    print_every = (num_steps / 6) // 1

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    # Shapes of the weight vectors
    w1_shape = [X_train.shape[1], n_units_1]
    w2_shape = [n_units_1, n_units_2]
    w3_shape = [n_units_2, 10]

    # Start with Xavier initialized weights (could be random with sigma of 0.01, but this works well too)
    initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float64)
    # Initialize all variables
    weights_1 = tf.Variable(initializer(w1_shape))
    weights_2 = tf.Variable(initializer(w2_shape))
    weights_3 = tf.Variable(initializer(w3_shape))
    # Initialize biases at 0: looked in literature, seems to be accepted and work well
    b_1 = tf.Variable(initial_value=np.zeros((n_units_1)),
                      name='b_1', dtype='float64')
    b_2 = tf.Variable(initial_value=np.zeros((n_units_2)),
                      name='b_2', dtype='float64')
    b_3 = tf.Variable(initial_value=np.zeros((10)),
                      name='b_3', dtype='float64')

    # For Dropout
    keep_prob = tf.placeholder(name='keep_prob', dtype='float64')

    # Create Placeholders for input x and y
    input_X = tf.placeholder(name='input_X', dtype='float64')
    input_y = tf.placeholder(name='input_y', dtype='float64')

    # Hidden layer 1 with relu activation
    logits_1 = tf.matmul(input_X, weights_1) + b_1
    relu_1 = tf.nn.relu(logits_1)
    # Hidden layer 2 with relu activation
    logits_2 = tf.matmul(relu_1, weights_2) + b_2
    relu_2 = tf.nn.relu(logits_2)
    # Try a dropout layer
    dropout = tf.nn.dropout(relu_2, keep_prob)
    # output layer (softmax activation for predictions)
    logits_3 = tf.matmul(dropout, weights_3) + b_3
    predicted_y = tf.nn.softmax(logits_3)
    # Use softmax cross entropy loss function, built in to tf
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=input_y, logits=logits_3))
    # Adam optimizer works, I achieved 98% 0.001
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    # In order to judge accuracy, compare predicted and actual y
    correct_prediction = tf.equal(
        tf.argmax(predicted_y, 1), tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Record the losses so that it can be plotted at the end
    train_loss_evolution = []
    val_loss_evolution = []
    with tf.Session() as sess:
        # Run the network:
        starttime = datetime.datetime.now()
        sess.run(tf.global_variables_initializer())
        for step in range(num_steps):
            # Minibatches are randomly selected
            batch_start = np.random.randint(len(X_train) - batch_size)
            # make the mini batch
            batch_x = X_train[batch_start:(batch_start + batch_size), :]
            batch_y = y_train[batch_start:(batch_start + batch_size), :]
            # Train on the batches that were just created
            sess.run(optimizer, {input_X: batch_x,
                                 input_y: batch_y, keep_prob: 0.2})
            # Record training and validation losses every 250 iterations:
            if step % record_every == 0:
                train_loss_i = sess.run(
                    loss, {input_X: batch_x, input_y: batch_y, keep_prob: 0.2})
                train_loss_evolution.append(train_loss_i)
                val_loss_i = sess.run(
                    loss, {input_X: X_val, input_y: y_val, keep_prob: 0.2})
                val_loss_evolution.append(val_loss_i)
            # print out and plot how things are going every once in a while
            if step % print_every == 0:
                print("Training loss at iter {}: {:.4f}".format(step, train_loss_i))
                print("Validation loss at iter {}, {:.4f}".format(step, val_loss_i))
                # Calculate and print the training accuracy and validation accuracy:
                train_acc = accuracy.eval(
                    feed_dict={input_X: X_train, input_y: y_train, keep_prob: 1.0})
                val_acc = accuracy.eval(
                    feed_dict={input_X: X_val, input_y: y_val, keep_prob: 1.0})
                print("Training set accuracy: {:.1f}%".format(train_acc * 100))
                print("Validation set accuracy: {:.1f}%".format(val_acc * 100))
                print()
                plt.close()
                x = [
                    i * record_every for i in list(range(len(val_loss_evolution)))]
                plt.plot(x, train_loss_evolution, '-',
                         label='Training Loss', color='blue')
                plt.plot(x, val_loss_evolution, '-',
                         label='Validation Loss', color='darkorange')
                plt.legend()
                plt.xlabel('Iteration')
                plt.ylabel('loss')
                plt.pause(0.000001)
        print()
        # Get accuracy score on the test set in the end
        test_acc = accuracy.eval(
            feed_dict={input_X: X_test, input_y: y_test, keep_prob: 1.0})
        print("Final Training Loss: {:.4f}".format(train_loss_i))
        print("Final Validation Loss: {:.4f}".format(val_loss_i))
        print("Final Test Set accuracy: {:.1f}%".format(test_acc * 100))
        training_time = datetime.datetime.strptime(
            str(datetime.datetime.now() - starttime), '%H:%M:%S.%f')
        print(training_time.strftime(
            'Training Took %H hours, %M minutes and %S seconds'))


model_2layer_dropout()
