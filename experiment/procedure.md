### Procedure

#### Model Selection
Two pretrained deep convolutional neural network models are used for transfer learning: **VGG19** and **MobileNetV2**. Select one model.

**1. VGG19:**
Deep CNN architecture with 19 layers that uses small convolution filters to learn rich and detailed image features.

**2. MobileNetV2:**
Lightweight and efficient CNN model designed for faster training and low-resource environments. It uses depth-wise separable convolutions and inverted residual blocks.

#### Freezing/Unfreezing Options
Based on the selected model, the following freezing options are available:

**MobileNetV2 Options:**
- **Base Freeze:** Freeze all backbone layers (0% unfreeze)
- **10% Unfreeze:** Unfreeze last 10% layers
- **20% Unfreeze:** Unfreeze last 20% layers

**VGG19 Options:**
- **Base Freeze:** Freeze all backbone layers (0% unfreeze)
- **20% Unfreeze:** Unfreeze last 20% layers
- **30% Unfreeze:** Unfreeze last 30% layers

---

#### Steps

**Step 1: Start the Experiment Setup**
Initialize the Python environment and import the required libraries such as TensorFlow/Keras, NumPy, Matplotlib, and other supporting packages for training and visualization.

**Step 2: Load the Dataset**
- Load the **Oxford Flowers** dataset and organize it into training and validation sets.
- Perform required preprocessing such as resizing images to a fixed input size supported by pretrained models (e.g., 224×224) and normalizing pixel values.

**Step 3: Apply Data Augmentation (If Used)**
Apply image augmentation techniques such as rotation, zoom, flipping, and shifting to improve model generalization and reduce overfitting on limited data.

**Step 4: Model Selection (Choose Backbone)**
Select one pretrained model backbone from the following options:
- **Option 1: VGG19**
- **Option 2: MobileNetV2**

Load the selected model using pretrained weights (commonly ImageNet weights) and exclude the original top classification layers (`include_top=False`). This is necessary because the original top layers are specific to ImageNet classes, which differ from the target dataset. Removing them allows the model to use pretrained features while adding a new classification head suitable for the current task.

**Step 5: Build the Classification Head**
Add new layers on top of the pretrained backbone to match the dataset output classes. Generally, the classification head includes:
- Global Average Pooling / Flatten layer
- Fully Connected (Dense) layer(s)
- Dropout (optional)
- Final Dense layer with Softmax activation for multi-class classification

**Step 6: Set Freezing / Unfreezing Strategy (Interactive Toggle)**
After selecting the model, choose the layer freezing option based on the model:

*For MobileNetV2:*
- **Base Freeze:** Freeze all backbone layers (feature extractor only)
- **10% Unfreeze:** Unfreeze only the last 10% layers of the backbone for fine-tuning
- **20% Unfreeze:** Unfreeze only the last 20% layers of the backbone for fine-tuning

*For VGG19:*
- **Base Freeze:** Freeze all backbone layers (feature extractor only)
- **20% Unfreeze:** Unfreeze only the last 20% layers of the backbone for fine-tuning
- **30% Unfreeze:** Unfreeze only the last 30% layers of the backbone for fine-tuning

**Step 7: Compile the Model**
Compile the model with appropriate settings such as:
- **Optimizer:** Adam / SGD
- **Loss function:** Categorical Cross-Entropy / Sparse Categorical Cross-Entropy
- **Metrics:** Accuracy

*Note: Use a lower learning rate when fine-tuning to avoid disturbing pretrained weights too aggressively.*

**Step 8: Train the Model with Limited Steps**
Train the model for a limited number of epochs/iterations as per experiment constraints. During training, monitor:
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

**Step 9: Plot Accuracy and Loss Snapshots**
After training, plot graphs for quick comparison:
- Accuracy vs Epochs
- Loss vs Epochs

This provides a clear understanding of model learning behavior under different freezing/unfreezing configurations.

**Step 10: Compare Results Across Configurations**
Compare all experimental results for:
- MobileNetV2: Base Freeze vs 10% vs 20% unfreeze.
- VGG19: Base Freeze vs 20% vs 30% unfreeze

Analyze which combination gives better accuracy and stable loss.

**Step 11: Final Conclusion**
Conclude which pretrained backbone and which unfreeze percentage performs best for the Oxford Flowers dataset, and summarize the observations based on training curves and validation performance.