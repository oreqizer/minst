#ifndef MNIST_ENUMS_H
#define MNIST_ENUMS_H

// Constants
#define EPSILON 0.0000001 // Adagrad / RMSProp prevents division by zero

// Layers
#define LAYER_IN 28 * 28                       // Input layer, image pixels
#define LAYER_IN_BIAS LAYER_IN + 1             // Input layer with bias
#define LAYER_HIDDEN_1 200                     // Hidden layer neurons
#define LAYER_HIDDEN_1_BIAS LAYER_HIDDEN_1 + 1 // Hidden layer with bias
#define LAYER_HIDDEN_2 80                      // Hidden layer neurons
#define LAYER_HIDDEN_2_BIAS LAYER_HIDDEN_2 + 1 // Hidden layer with bias
#define LAYER_OUT 10                           // 0..9

// Hyperparameters
#define EPOCHS 10           // Iterations over dataset
#define BATCH 100           // Number of images to adjust weights after
#define BIAS 1              // Bias value
#define WEIGHT_INIT 0.1     // Random init bounds interval -WEIGHT_INIT..WEIGHT_INIT
#define RATE 0.1            // Learning rate
#define DECAY 0.0001        // Decay of learning rate after update
#define MOMENTUM 0.9        // How much does previous cache affect the next one

#endif //MNIST_ENUMS_H
