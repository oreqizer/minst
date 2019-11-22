#ifndef MNIST_ENUMS_H
#define MNIST_ENUMS_H

// Constants
#define EPSILON 0.0000001     // Adagrad / RMSProp prevents division by zero

// Layers
#define LAYER_1 28 * 28       // Image pixels
#define LAYER_2 LAYER_1 / 8   // Arbitrary
#define LAYER_3 10            // 0..9

// Hyperparameters
#define EPOCHS 15             // Iterations over dataset
#define BATCH 200             // Number of images to adjust weights after
#define RATE 0.15             // Learning rate
#define RHO 0.9               // How much does previous cache affect the next one

#endif //MNIST_ENUMS_H
