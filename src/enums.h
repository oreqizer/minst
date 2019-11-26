#ifndef MNIST_ENUMS_H
#define MNIST_ENUMS_H

// Constants
#define EPSILON 0.0000001 // Adagrad / RMSProp prevents division by zero
#define EPSILON_INIT 0.12 // Random init bounds interval -EPSILON_INIT..EPSILON_INIT

// Layers
#define LAYER_IN 28 * 28                   // Input layer, image pixels
#define LAYER_IN_BIAS LAYER_IN + 1         // Input layer with bias
#define LAYER_HIDDEN 64                    // Hidden layer neurons
#define LAYER_HIDDEN_BIAS LAYER_HIDDEN + 1 // Hidden layer with bias
#define LAYER_OUT 10                       // 0..9

// Hyperparameters
#define EPOCHS 5  // Iterations over dataset
#define BATCH 200 // Number of images to adjust weights after
#define RATE 0.3  // Learning rate
#define RHO 0.9   // How much does previous cache affect the next one

#endif //MNIST_ENUMS_H
