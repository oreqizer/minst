#ifndef MNIST_ENUMS_H
#define MNIST_ENUMS_H

// Constants
#define EPSILON 0.0000001 // Adagrad / RMSProp prevents division by zero

// Layers
#define LAYER_IN 2                             // Input layer, image pixels
#define LAYER_IN_BIAS LAYER_IN + 1             // Input layer with bias
#define LAYER_HIDDEN_1 4                       // Hidden layer neurons
#define LAYER_HIDDEN_1_BIAS LAYER_HIDDEN_1 + 1 // Hidden layer with bias
#define LAYER_OUT 1                            // 0..9

// Hyperparameters
#define EPOCHS 5000       // Iterations over dataset
#define BATCH 1           // Number of images to adjust weights after
#define BIAS 1            // Bias value
#define WEIGHT_INIT 0.15  // Random init bounds interval -WEIGHT_INIT..WEIGHT_INIT
#define RATE 0.8          // Learning rate
#define DECAY 0.00002     // Decay of learning rate after update
#define MOMENTUM 0.9      // How much does previous cache affect the next one

#endif //MNIST_ENUMS_H
