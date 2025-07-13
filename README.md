# OB-GAN: Object-Based Generative Adversarial Network

## Overview

OB-GAN is a novel Generative Adversarial Network (GAN) architecture designed specifically for medical image generation, particularly focused on lung nodule detection and generation. The model uses a dual-generator approach with an object detection-based discriminator to generate realistic lung images with synthetic nodules.

## Architecture

### Dual Generator System

The OB-GAN employs two specialized generators:

1. **Lung Generator (`GeneratorLung`)**: 
   - Generates complete lung images (1024x1024)
   - Uses transposed convolutions with batch normalization and ReLU activation
   - Input: 27x27x27 noise vector
   - Output: 1-channel grayscale lung image

2. **Nodule Generator (`GeneratorNodule`)**:
   - Generates individual nodule patches (54x54)
   - Smaller architecture optimized for nodule generation
   - Input: 27x27x27 noise vector
   - Output: 1-channel grayscale nodule image

### Object Detection Discriminator

The discriminator is based on **Faster R-CNN** architecture:
- Pre-trained on COCO dataset
- Modified for binary classification (background vs. nodule)
- Provides bounding box detection capabilities
- Enables the GAN to learn spatial relationships and object localization


## Dataset Requirements

The model expects a dataset with the following structure:
```
FinalDataset/
├── train/
│   ├── image1.jpg
│   ├── image1.xml
│   ├── image2.jpg
│   ├── image2.xml
│   └── ...
├── valid/
│   ├── image1.jpg
│   ├── image1.xml
│   └── ...
└── test/
    ├── image1.jpg
    ├── image1.xml
    └── ...
```

### Annotation Format
XML files should contain Pascal VOC format annotations:
```xml
<annotation>
    <object>
        <name>Nodule</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>
```

## Installation

1. Clone the repository:
```bash
git clone [<repository-url>](https://github.com/OmGuin/OB-GAN1)
cd OB-GAN1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your dataset is properly organized in the `FinalDataset/` directory.

## Usage

### Training

To train the OB-GAN model:

```bash
python OB-GAN.py
```

Training parameters can be modified in `config.py`:
- `BATCH_SIZE`: Training batch size (default: 8)
- `NUM_EPOCHS`: Number of training epochs (default: 15)
- `RESIZE_TO`: Image resolution (default: 1024)
- `NUM_WORKERS`: Data loading workers (default: 4)

### Testing

To test the generators and generate sample images:

```bash
python test_external.py
```

The testing script will:
- Generate 16 sample images using both generators
- Save individual images and grid visualizations
- Calculate and save statistical analysis
- Output results to `modelandplots/` directory

### Model Loading

To use a trained model for testing, modify the `model_path` variable in `test_external.py`:

```python
model_path = 'modelandplots/best_model.pth'  # or 'modelandplots/last_model.pth'
```

## Output Files

### Training Outputs
- `modelandplots/best_model.pth`: Best model checkpoint
- `modelandplots/last_model.pth`: Last epoch checkpoint
- `modelandplots/train_loss.png`: Training loss plots
- `modelandplots/valid_loss.png`: Validation loss plots

### Testing Outputs
- `modelandplots/generated_lung_*.png`: Individual generated lung images
- `modelandplots/generated_nodule_*.png`: Individual generated nodule images
- `modelandplots/generated_lung_column1.png`: Sample 8 lung images in single column
- `modelandplots/generated_lung_column2.png`: Sample 8 lung images in single column
- `modelandplots/generated_nodule_column1.png`: Sample 8 nodule images in single column
- `modelandplots/generated_nodule_column2.png`: Sample 8 nodule images in single column
- `modelandplots/generator_test_statistics.json`: Detailed statistics
- `modelandplots/test_summary.txt`: Human-readable summary

## Model Architecture Details

### Generator Architectures

**Lung Generator**:
```
Input (27x27x27) → ConvTranspose2d → BatchNorm → ReLU → ... → Sigmoid → Output (1024x1024)
```

**Nodule Generator**:
```
Input (27x27x27) → ConvTranspose2d → BatchNorm → ReLU → ConvTranspose2d → Sigmoid → Output (54x54)
```

### Loss Functions

The model uses three loss components:
1. **Lung Generator Loss**: Binary cross-entropy between generated and real lung images
2. **Nodule Generator Loss**: Binary cross-entropy between generated nodules and target nodule characteristics
3. **Discriminator Loss**: Object detection loss from Faster R-CNN

## Performance Metrics

The testing framework provides comprehensive statistics:
- **Image Statistics**: Mean, standard deviation, min/max values
- **Generation Time**: Average time per image generation
- **Quality Metrics**: Pixel value distributions and characteristics

## Technical Requirements

- **Python**: 3.7+
- **PyTorch**: 1.9.0+
- **CUDA**: Recommended for GPU acceleration
- **Memory**: Minimum 8GB RAM, 16GB+ recommended
- **Storage**: Sufficient space for dataset and model checkpoints

## Acknowledgments

- Faster R-CNN implementation from torchvision
- Medical image processing techniques from the research community
- Dataset preprocessing utilities and augmentation techniques
