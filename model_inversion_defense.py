"""
Author: Raghuram Venkatapuram
CIS6372 - Information Assurance

Model Inversion Defense Mechanisms - Project
"""

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import f1_score

# Base LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # Implement forward pass with ReLu activation
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Defense 1: Gradient Clipping
class LeNetGradientClip(LeNet):
    def __init__(self, clip_value=1.0):
        super(LeNetGradientClip, self).__init__()
        self.clip_value = clip_value

    # Implement forward pass with ReLu activation
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Gradient Clipping
    def clip_gradients(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.clip_value, self.clip_value)

# Defense 2: Differential Privacy
class LeNetDifferentialPrivacy(LeNet):
    def __init__(self, noise_scale=0.1):
        super(LeNetDifferentialPrivacy, self).__init__()
        self.noise_scale = noise_scale

    # Implement forward pass with ReLu activation
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Adding noise to gradients
    def add_noise(self):
        for param in self.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * self.noise_scale
                param.grad.data += noise

# Defense 3: Adversarial Training
class LeNetAdversarial(LeNet):
    def __init__(self, epsilon=0.1):
        super(LeNetAdversarial, self).__init__()
        self.epsilon = epsilon

    # Implement forward pass with ReLu activation
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # Generating adversarial attack examples
    def generate_adversarial_example(self, x, y):
        x.requires_grad = True
        output = self(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        perturbation = self.epsilon * torch.sign(x.grad.data)
        x_adv = x.data + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv

# Defense 4: Knowledge Distillation
class LeNetDistilled(LeNet):
    def __init__(self, temperature=2.0):
        super(LeNetDistilled, self).__init__()
        self.temperature = temperature

    # Implement forward pass with ReLu activation
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x) / self.temperature
        return x

# iDLG Attack
class iDLGAttack:
    def __init__(self, model, device='metal'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.hook_handles = []
        self.last_layer_grad = None
        self._register_hooks()

    # Generating hooks for layers
    def _register_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            self.last_layer_grad = grad_output[0].detach()
        
        # Register hook for the last layer
        self.hook_handles.append(self.model.fc3.register_backward_hook(hook_fn))

    # Label inference logic
    def infer_label(self, gradient):
        label = torch.argmin(torch.sum(gradient, dim=0))
        return label

    # Obtaining gradients from true labels (inference)
    def get_gradients(self, images, labels=None):
        self.model.zero_grad()
        outputs = self.model(images)
        
        if labels is None:
            # Use dummy label for initial backward pass
            dummy_labels = torch.zeros_like(outputs)
            dummy_labels[:, 0] = 1
            loss = self.criterion(outputs, dummy_labels.argmax(dim=1))
        else:
            loss = self.criterion(outputs, labels)
            
        loss.backward()
        
        # Get gradients
        gradients = []
        for param in self.model.parameters():
            if param.requires_grad:
                gradients.append(param.grad.clone())
        
        # Infer true label if not provided
        if labels is None:
            inferred_label = self.infer_label(self.last_layer_grad)
            return gradients, inferred_label
        return gradients

    # Reconstruction of data
    def reconstruct_data(self, original_gradients, num_iterations=1000, lr=0.1):
        dummy_data = torch.randn(1, 1, 28, 28).to(self.device)
        _, inferred_label = self.get_gradients(dummy_data)
        dummy_data = torch.randn(1, 1, 28, 28, device=self.device).requires_grad_(True)
        
        optimizer = torch.optim.Adam([dummy_data], lr=lr)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            self.model.zero_grad()
            
            outputs = self.model(dummy_data)
            loss = self.criterion(outputs, inferred_label.unsqueeze(0))
            loss.backward(retain_graph=True)
            
            dummy_gradients = []
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    dummy_gradients.append(param.grad.clone())
            
            # Compute gradient difference
            grad_diff = torch.tensor(0., device=self.device, requires_grad=True)
            for dg, og in zip(dummy_gradients, original_gradients):
                grad_diff = grad_diff + ((dg - og) ** 2).sum()
            
            # Update using the gradient difference
            grad_diff.backward()
            optimizer.step()
            
            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, 0, 1)
            
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Gradient Difference: {grad_diff.item():.4f}')
        
        return dummy_data

# DLG Attack
class DLGAttack:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.feature_maps = []
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture intermediate feature maps"""
        def hook_fn(module, input, output):
            self.feature_maps.append(output)

        # Register hooks for conv and fc layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_forward_hook(hook_fn)
                self.hook_handles.append(handle)

    def _clear_features(self):
        """Clear stored feature maps"""
        self.feature_maps = []

    def get_gradients(self, images, labels):
        """Get gradients from the model"""
        self.model.zero_grad()
        self._clear_features()
        
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        
        gradients = []
        for param in self.model.parameters():
            if param.requires_grad:
                gradients.append(param.grad.clone())
        
        return gradients, self.feature_maps

    # Total variation loss for regularization
    def total_variation_loss(self, x):
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        return torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))

    # Reconstruction of data
    def reconstruct_data(self, original_gradients, target_label, num_iterations=1000, lr=0.1):
        original_grads, original_features = original_gradients
        
        # Initialize with random noise
        dummy_data = torch.randn(1, 1, 28, 28, device=self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([dummy_data], lr=lr)
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            self.model.zero_grad()
            self._clear_features()
            
            outputs = self.model(dummy_data)
            loss = self.criterion(outputs, torch.tensor([target_label]).to(self.device))
            loss.backward()
            
            dummy_gradients = []
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    dummy_gradients.append(param.grad.clone())
            
            # Compute gradient difference
            grad_diff = torch.tensor(0., device=self.device, requires_grad=True)
            for dg, og in zip(dummy_gradients, original_grads):
                grad_diff = grad_diff + ((dg - og) ** 2).sum()
            
            # Add TV regularization
            tv_loss = self.total_variation_loss(dummy_data)
            total_loss = grad_diff + 0.1 * tv_loss
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, 0, 1)
            
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss: {total_loss.item():.4f}')
        
        return dummy_data

# Training the model
def train_model(model, train_loader, test_loader, epochs=10, defense_type=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_losses = []
    test_accuracies = []
    test_f1_scores = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            
            # Apply defense-specific training
            if defense_type == 'adversarial':
                # Generate adversarial examples
                inputs_adv = model.generate_adversarial_example(inputs, labels)
                outputs = model(inputs_adv)
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply defense-specific gradient modifications
            if defense_type == 'gradient_clip':
                model.clip_gradients()
            elif defense_type == 'differential_privacy':
                model.add_noise()
            
            optimizer.step()
            running_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        test_f1_scores.append(f1)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}')
    
    return train_losses, test_accuracies, test_f1_scores

# Model Inversion Attack
def model_inversion_attack(model, target_class, steps=1000, lr=0.1):
    # Initialize a random image
    x = torch.randn(1, 1, 28, 28, requires_grad=True)
    optimizer = optim.Adam([x], lr=lr)
    
    target = torch.tensor([target_class])
    
    for step in range(steps):
        optimizer.zero_grad()
        output = model(x)
        loss = -output[0, target_class]  # Maximize the target class probability
        loss.backward()
        optimizer.step()
        
        # Clamp values to valid image range
        x.data = torch.clamp(x.data, 0, 1)
    
    return x.detach()

def evaluate_defenses():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    # Get a sample '0' digit from MNIST for comparison
    target_class = 0
    original_image = None
    for images, labels in testloader:
        for i in range(len(labels)):
            if labels[i] == target_class:
                original_image = images[i:i+1]
                break
        if original_image is not None:
            break
    
    # Initialize models
    models = {
        'baseline': LeNet(),
        'gradient_clip': LeNetGradientClip(clip_value=1.0),
        'differential_privacy': LeNetDifferentialPrivacy(noise_scale=0.1),
        'adversarial': LeNetAdversarial(epsilon=0.1),
        'distilled': LeNetDistilled(temperature=2.0)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        train_losses, test_accuracies, test_f1_scores = train_model(
            model, trainloader, testloader, 
            epochs=10, 
            defense_type=name if name != 'baseline' else None
        )
        results[name] = {
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'test_f1_scores': test_f1_scores,
            'final_accuracy': test_accuracies[-1],
            'final_f1': test_f1_scores[-1],
            'final_loss': train_losses[-1]
        }
    
    # Perform model inversion attacks
    attack_results = {}
    dlg_attack_results = {}
    idlg_attack_results = {}
    for name, model in models.items():
        print(f"\nPerforming model inversion attack on {name} model...")
        reconstructed = model_inversion_attack(model, target_class)
        attack_results[name] = reconstructed
        
        print(f"\nPerforming DLG attack on {name} model...")
        dlg = DLGAttack(model)
        dlg_gradients = dlg.get_gradients(original_image, torch.tensor([target_class]))
        dlg_reconstructed = dlg.reconstruct_data(dlg_gradients, target_class)
        dlg_attack_results[name] = dlg_reconstructed
        
        print(f"\nPerforming iDLG attack on {name} model...")
        idlg = iDLGAttack(model)
        idlg_gradients = idlg.get_gradients(original_image, torch.tensor([target_class]))
        idlg_reconstructed = idlg.reconstruct_data(idlg_gradients)
        idlg_attack_results[name] = idlg_reconstructed
    
    # Calculate reconstruction metrics
    reconstruction_metrics = {}
    dlg_metrics = {}
    idlg_metrics = {}
    for name, reconstructed in attack_results.items():
        if original_image is not None:
            # Calculate MSE between original and reconstructed
            mse = torch.mean((original_image - reconstructed) ** 2).item()
            # Calculate SSIM (Structural Similarity Index)
            original_np = original_image[0, 0].cpu().numpy()
            reconstructed_np = reconstructed[0, 0].cpu().numpy()
            ssim_score = ssim(original_np, reconstructed_np, data_range=2.0)
            reconstruction_metrics[name] = {
                'mse': mse,
                'ssim': ssim_score
            }
            
            # Calculate metrics for DLG attack
            dlg_reconstructed = dlg_attack_results[name]
            dlg_mse = torch.mean((original_image - dlg_reconstructed) ** 2).item()
            dlg_reconstructed_np = dlg_reconstructed[0, 0].detach().cpu().numpy()
            dlg_ssim = ssim(original_np, dlg_reconstructed_np, data_range=2.0)
            dlg_metrics[name] = {
                'mse': dlg_mse,
                'ssim': dlg_ssim
            }
            
            # Calculate metrics for iDLG attack
            idlg_reconstructed = idlg_attack_results[name]
            idlg_mse = torch.mean((original_image - idlg_reconstructed) ** 2).item()
            idlg_reconstructed_np = idlg_reconstructed[0, 0].detach().cpu().numpy()
            idlg_ssim = ssim(original_np, idlg_reconstructed_np, data_range=2.0)
            idlg_metrics[name] = {
                'mse': idlg_mse,
                'ssim': idlg_ssim
            }
    
    # Save performance statistics to a text file
    with open('performance_stats.txt', 'w') as f:
        f.write("Model Performance Statistics\n")
        
        f.write("Training Results:\n")
        f.write("----------------\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"Final Accuracy: {result['final_accuracy']:.2f}%\n")
            f.write(f"Final F1 Score: {result['final_f1']:.4f}\n")
            f.write(f"Final Loss: {result['final_loss']:.4f}\n")
        
        f.write("\nReconstruction Metrics (Basic Attack):\n")
        f.write("-----------------------------------\n")
        for name, metrics in reconstruction_metrics.items():
            f.write(f"\n{name}:\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            
        f.write("\nReconstruction Metrics (DLG Attack):\n")
        f.write("-----------------------------------\n")
        for name, metrics in dlg_metrics.items():
            f.write(f"\n{name}:\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
            
        f.write("\nReconstruction Metrics (iDLG Attack):\n")
        f.write("-----------------------------------\n")
        for name, metrics in idlg_metrics.items():
            f.write(f"\n{name}:\n")
            f.write(f"MSE: {metrics['mse']:.4f}\n")
            f.write(f"SSIM: {metrics['ssim']:.4f}\n")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot training losses
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['train_losses'], label=name)
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracies
    plt.subplot(1, 3, 2)
    for name, result in results.items():
        plt.plot(result['test_accuracies'], label=name)
    plt.title('Test Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot test F1 scores
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        plt.plot(result['test_f1_scores'], label=name)
    plt.title('Test F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('defense_comparison.png')
    
    # Plot reconstructed images
    plt.figure(figsize=(20, 9))
    
    # Plot original image
    plt.subplot(3, len(attack_results) + 1, 1)
    if original_image is not None:
        plt.imshow(original_image[0, 0].cpu().numpy(), cmap='gray')
    plt.title('Original MNIST')
    plt.axis('off')
    
    # Plot basic attack reconstructions
    for i, (name, img) in enumerate(attack_results.items()):
        plt.subplot(3, len(attack_results) + 1, i + 2)
        plt.imshow(img[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title(f'{name}\nMSE: {reconstruction_metrics[name]["mse"]:.4f}\nSSIM: {reconstruction_metrics[name]["ssim"]:.4f}')
        plt.axis('off')
    
    # Plot DLG attack reconstructions
    plt.subplot(3, len(attack_results) + 1, len(attack_results) + 2)
    if original_image is not None:
        plt.imshow(original_image[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title('Original MNIST')
    plt.axis('off')
    
    for i, (name, img) in enumerate(dlg_attack_results.items()):
        plt.subplot(3, len(attack_results) + 1, len(attack_results) + 3 + i)
        plt.imshow(img[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title(f'{name} (DLG)\nMSE: {dlg_metrics[name]["mse"]:.4f}\nSSIM: {dlg_metrics[name]["ssim"]:.4f}')
        plt.axis('off')
    
    # Plot iDLG attack reconstructions
    plt.subplot(3, len(attack_results) + 1, 2 * (len(attack_results) + 1) + 1)
    if original_image is not None:
        plt.imshow(original_image[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title('Original MNIST')
    plt.axis('off')
    
    for i, (name, img) in enumerate(idlg_attack_results.items()):
        plt.subplot(3, len(attack_results) + 1, 2 * (len(attack_results) + 1) + 2 + i)
        plt.imshow(img[0, 0].detach().cpu().numpy(), cmap='gray')
        plt.title(f'{name} (iDLG)\nMSE: {idlg_metrics[name]["mse"]:.4f}\nSSIM: {idlg_metrics[name]["ssim"]:.4f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstructed_images.png')

if __name__ == "__main__":
    evaluate_defenses() 