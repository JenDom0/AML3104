Assignment 4: MNIST Digit Classification Using Neural Network

Dataset Used:  MNIST – A collection of 70,000 grayscale images of handwritten digits (0–9), each image is 28x28 pixels.
•	Training set: 60,000 images
•	Test set: 10,000 images
•	Readily accessible via libraries like TensorFlow or PyTorch. 

1. Model Architecture & Rationale
I chose a Convolutional Neural Network (CNN) because its shared 3×3 filters excel at learning local stroke and edge patterns directly from 2D images, and max pooling adds shift invariance so small translations don’t derail the model. My network is kept lightweight for fast training:
•	Input: 28×28×1 grayscale images
•	Conv Block 1:
o	Conv2D(32,3×3)+ReLU
o	BatchNormalization
o	MaxPooling2D(2×2)
o	Dropout(0.25)
•	Conv Block 2: same pattern with 64 filters
•	Classifier Head:
o	Flatten → Dense(128,ReLU) + Dropout(0.5)
o	Dense(10,Softmax)
Batch norm speeds convergence and dropout prevents overfitting, while two shallow blocks capture both fine and mid level handwriting features without excessive compute.

2. Preprocessing Steps
•	Normalization: scale pixel values to [0, 1] for stable gradients
•	Reshaping: add channel dimension (-1,28,28,1) for Conv2D
•	Validation Split: reserve 10% of training data (validation_split=0.1)
•	Callbacks:
o	EarlyStopping (patience 3) to halt when validation loss stalls
o	ReduceLROnPlateau to cut learning rate by 50% after 2 stagnant epochs

3. Test Set Performance
•	Training Accuracy: 99.5%
•	Test Accuracy: 99.2%
•	Gap: 0.3 percentage points → minimal overfitting, excellent generalization.

4. Misclassified Examples
True	Predicted	Likely Cause
5	3	Open top “5” resembles a “3”
7	1	Slanted “7” without crossbar looks like a “1”
9	4	Small loop and long tail on “9” mimic a “4”
Insight: Errors occur when handwriting deviates from canonical shapes, causing class overlap in pixel space.

5. Further Improvements
•	Data Augmentation: rotations, shifts, zooms to cover more handwriting styles
•	Deeper Architecture: add a third conv block or residual connections
•	Advanced Regularization: Mixup/Cutout and label smoothing for robust features
•	Ensembling: average predictions from multiple CNNs to smooth out mistakes
•	Hyperparameter Tuning: refine filter counts, dropout rates, and learning rate schedules
Bottom line: The two block CNN already achieves ~99.2% on MNIST; these tweaks can push it even closer to perfect.
