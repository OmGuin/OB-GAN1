import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch.utils.data import DataLoader
import json
from datetime import datetime
import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
import importlib.util
from datasets import CustomDataset
from custom_utils import get_valid_transform, collate_fn
from config import DEVICE, RESIZE_TO, CLASSES, NUM_CLASSES, OUT_DIR

spec = importlib.util.spec_from_file_location("ob_gan", os.path.join(os.path.dirname(__file__), "OB-GAN.py"))
if spec is None or spec.loader is None:
    raise ImportError("Could not load OB-GAN.py as a module. Please check the file name and location.")
ob_gan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ob_gan)

class GANGeneratorTester:
    def __init__(self, model_path=None, test_dir='FinalDataset/test', output_dir=OUT_DIR):
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.device = DEVICE
                
        self.generator_lung = ob_gan.GeneratorLung(z_dim=27, channels_img=1, features_g=32)
        self.generator_nodule = ob_gan.GeneratorNodule(z_dim=27, channels_img=1, features_g=32)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No model path provided or model not found. Using untrained generators.")
        
        self.generator_lung.to(self.device)
        self.generator_nodule.to(self.device)
        
        self.generator_lung.eval()
        self.generator_nodule.eval()
        
        self.test_dataset = CustomDataset(
            test_dir, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=collate_fn
        )
        
        self.stats = {
            'test_images_processed': 0,
            'lung_generator_stats': {},
            'nodule_generator_stats': {},
            'generation_times': [],
            'test_timestamp': datetime.now().isoformat()
        }
    
    def load_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # handle cases where no model is provided
            if 'model_state_dict' in checkpoint:
                gan_model = ob_gan.GAN(num_classes=NUM_CLASSES)
                gan_model.load_state_dict(checkpoint['model_state_dict'])
                
                self.generator_lung.load_state_dict(gan_model.generatorlung.state_dict())
                self.generator_nodule.load_state_dict(gan_model.generatornodule.state_dict())
                
                print(f"Successfully loaded model from {model_path}")
                print(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
            else:
                print("Checkpoint doesn't contain model_state_dict")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained generators")
    
    def generate_images(self, num_images=16):
        print(f"Generating {num_images} images...")
        
        generated_lung_images = []
        generated_nodule_images = []
        
        with torch.no_grad():
            for i in range(num_images):
                # gaussian niose
                z = torch.randn(1, 27, 27, 27).to(self.device)
                
                start_time = time.time()
                lung_image = self.generator_lung(z)
                nodule_image = self.generator_nodule(z)
                end_time = time.time()
                generation_time = (end_time - start_time) * 1000  # ms
                self.stats['generation_times'].append(generation_time)
                
                generated_lung_images.append(lung_image.cpu())
                generated_nodule_images.append(nodule_image.cpu())
                
                if (i + 1) % 4 == 0:
                    print(f"Generated {i + 1}/{num_images} images")
        
        return generated_lung_images, generated_nodule_images
    
    def analyze_generated_images(self, lung_images, nodule_images):
        print("Analyzing generated images...")
        
        # move off gpu cause numpy needs cpu
        lung_np = torch.cat(lung_images, dim=0).squeeze().numpy()
        nodule_np = torch.cat(nodule_images, dim=0).squeeze().numpy()
        
        # Lung image statistics
        self.stats['lung_generator_stats'] = {
            'mean': float(np.mean(lung_np)),
            'std': float(np.std(lung_np)),
            'min': float(np.min(lung_np)),
            'max': float(np.max(lung_np)),
            'shape': lung_np.shape
        }
        
        # Nodule image statistics
        self.stats['nodule_generator_stats'] = {
            'mean': float(np.mean(nodule_np)),
            'std': float(np.std(nodule_np)),
            'min': float(np.min(nodule_np)),
            'max': float(np.max(nodule_np)),
            'shape': nodule_np.shape
        }
        
        # Generation time statistics
        if self.stats['generation_times']:
            self.stats['generation_time_stats'] = {
                'mean_time_ms': float(np.mean(self.stats['generation_times'])),
                'std_time_ms': float(np.std(self.stats['generation_times'])),
                'min_time_ms': float(np.min(self.stats['generation_times'])),
                'max_time_ms': float(np.max(self.stats['generation_times']))
            }
        
    
    def save_generated_images(self, lung_images, nodule_images, num_images=16):
        print(f"Saving {num_images} generated images...")
        
        lung_grid = torch.cat(lung_images[:num_images], dim=0)
        nodule_grid = torch.cat(nodule_images[:num_images], dim=0)
        
        for i in range(min(num_images, len(lung_images))):
            lung_path = os.path.join(self.output_dir, f'generated_lung_{i+1:02d}.png')
            save_image(lung_images[i], lung_path, normalize=True)
            
            nodule_path = os.path.join(self.output_dir, f'generated_nodule_{i+1:02d}.png')
            save_image(nodule_images[i], nodule_path, normalize=True)
        
        # 8 sample lung images
        lung_images_1_8 = []
        for i in range(min(8, len(lung_images))):
            img = lung_images[i].squeeze().numpy()
            lung_images_1_8.append(img)
        

        lung_column1 = np.vstack(lung_images_1_8)
        

        plt.figure(figsize=(4, 32))
        plt.imshow(lung_column1, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'generated_lung_column1.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
     
        if len(lung_images) >= 8:
            lung_images_9_16 = []
            for i in range(8, min(16, len(lung_images))):
                img = lung_images[i].squeeze().numpy()
                lung_images_9_16.append(img)
            

            lung_column2 = np.vstack(lung_images_9_16)
            

            plt.figure(figsize=(4, 32))
            plt.imshow(lung_column2, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'generated_lung_column2.png'), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        #  8 sample nodule images 
        nodule_images_1_8 = []
        for i in range(min(8, len(nodule_images))):
            img = nodule_images[i].squeeze().numpy()
            nodule_images_1_8.append(img)
        
        
        nodule_column1 = np.vstack(nodule_images_1_8)
        
        plt.figure(figsize=(4, 32))
        plt.imshow(nodule_column1, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'generated_nodule_column1.png'), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        if len(nodule_images) >= 8:
            nodule_images_9_16 = []
            for i in range(8, min(16, len(nodule_images))):
                img = nodule_images[i].squeeze().numpy()
                nodule_images_9_16.append(img)
            
            nodule_column2 = np.vstack(nodule_images_9_16)
            
            plt.figure(figsize=(4, 32))
            plt.imshow(nodule_column2, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'generated_nodule_column2.png'), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        print(f"Images saved to {self.output_dir}")
    
    def save_statistics(self):
        #json file for stats
        stats_path = os.path.join(self.output_dir, 'generator_test_statistics.json')
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=4)
        
        print(f"Statistics saved to {stats_path}")
        
        # Also  a human-readable summary
        summary_path = os.path.join(self.output_dir, 'test_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("GAN Generator Test Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {self.stats['test_timestamp']}\n")
            f.write(f"Device Used: {self.device}\n\n")
            
            f.write("Lung Generator Statistics:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.stats['lung_generator_stats'].items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nNodule Generator Statistics:\n")
            f.write("-" * 30 + "\n")
            for key, value in self.stats['nodule_generator_stats'].items():
                f.write(f"{key}: {value}\n")
            
            if 'generation_time_stats' in self.stats:
                f.write("\nGeneration Time Statistics (ms):\n")
                f.write("-" * 30 + "\n")
                for key, value in self.stats['generation_time_stats'].items():
                    f.write(f"{key}: {value:.2f}\n")
        
        print(f"Summary saved to {summary_path}")
    
    def run_test(self, num_images=16, model_path=None):
        print("Starting GAN Generator Test")
        print("=" * 40)
        
        if model_path:
            self.load_model(model_path)
        
        lung_images, nodule_images = self.generate_images(num_images)
        
        self.analyze_generated_images(lung_images, nodule_images)
        
        self.save_generated_images(lung_images, nodule_images, num_images)
        
        self.save_statistics()
        
        print("\nTest completed successfully!")
        print(f"Results saved to: {self.output_dir}")

def main():
    # model_path = 'modelandplots/best_model.pth'  # or 'modelandplots/last_model.pth'
    model_path = None  # Set to None to use untrained generators
    
    tester = GANGeneratorTester(
        model_path=model_path,
        test_dir='FinalDataset/test',
        output_dir=OUT_DIR
    )
    
    tester.run_test(num_images=16, model_path=model_path)

if __name__ == "__main__":
    main()
