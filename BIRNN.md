
1. Import Libraries

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

2. Load Data

We will load the training challenges, training solutions, and test challenges from the provided JSON files.

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Paths to data files (adjusted for Kaggle environment)
train_challenges_path = '/kaggle/input/arc-prize-2024/arc-agi_training_challenges.json'
train_solutions_path = '/kaggle/input/arc-prize-2024/arc-agi_training_solutions.json'
test_challenges_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'

# Load data
train_challenges = load_json_data(train_challenges_path)
train_solutions = load_json_data(train_solutions_path)
test_challenges = load_json_data(test_challenges_path)

3. Visualization Functions

We define functions to visualize the tasks and predictions.

# Define color map for visualization
cmap = colors.ListedColormap([
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#B10DC9'
])
norm = colors.Normalize(vmin=0, vmax=9)

# Function to plot input, output, and prediction grids
def plot_task(input_grid, output_grid=None, prediction_grid=None, title=''):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title, fontsize=16)
    grids = [input_grid, output_grid, prediction_grid]
    labels = ['Input', 'Output', 'Prediction']
    for ax, grid, label in zip(axs, grids, labels):
        if grid is not None:
            ax.imshow(grid, cmap=cmap, norm=norm)
        else:
            ax.axis('off')
        ax.set_title(label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

4. Define the Dataset Class

We create a custom dataset class ARCDataset to handle the ARC data.

# Custom Dataset class with data augmentation
class ARCDataset(Dataset):
    def __init__(self, challenges=None, solutions=None, augment=False, samples=None):
        self.samples = []
        self.augment = augment
        if samples is not None:
            self.samples = samples
        elif challenges is not None:
            for task_id, task in challenges.items():
                if solutions:
                    for idx, sample in enumerate(task['train']):
                        input_grid = np.array(sample['input'])
                        output_grid = np.array(sample['output'])
                        self.samples.append({
                            'input': input_grid,
                            'output': output_grid
                        })
                        if self.augment:
                            self.samples.extend(self.augment_sample(input_grid, output_grid))
                    if task_id in solutions:
                        for idx, sample in enumerate(task['test']):
                            input_grid = np.array(sample['input'])
                            output_grid = np.array(solutions[task_id][idx])
                            self.samples.append({
                                'input': input_grid,
                                'output': output_grid
                            })
                else:
                    for sample in task['test']:
                        input_grid = np.array(sample['input'])
                        self.samples.append({
                            'input': input_grid,
                            'output': None
                        })
        else:
            raise ValueError("Either challenges or samples must be provided.")

    def augment_sample(self, input_grid, output_grid):
        augmented_samples = []
        k = random.choice([0, 1, 2, 3])
        aug_input = np.rot90(input_grid, k).copy()
        aug_output = np.rot90(output_grid, k).copy()
        if random.choice([True, False]):
            aug_input = np.fliplr(aug_input).copy()
            aug_output = np.fliplr(aug_output).copy()
        if random.choice([True, False]):
            aug_input = np.flipud(aug_input).copy()
            aug_output = np.flipud(aug_output).copy()
        augmented_samples.append({'input': aug_input, 'output': aug_output})
        return augmented_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_grid = torch.tensor(sample['input'], dtype=torch.long)
        output_grid = sample['output']
        if output_grid is not None:
            output_grid = torch.tensor(output_grid, dtype=torch.long)
        return input_grid, output_grid

5. Define the Model

We use a UNet model architecture with LeakyReLU activations and dropout.

# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, num_classes=10):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(num_classes, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # Decoder
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.out_conv = nn.Conv2d(128, num_classes, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)  # Shape: [B, 64, H, W]
        x2 = self.enc_conv2(x1)  # Shape: [B, 128, H, W]
        x3 = self.enc_conv3(x2)  # Shape: [B, 256, H, W]
        # Decoder
        x = self.dec_conv1(x3)  # Shape: [B, 128, H, W]
        x = torch.cat([x, x2], dim=1)  # Concatenate along channels
        x = self.dropout(x)
        x = self.dec_conv2(x)  # Shape: [B, 64, H, W]
        x = torch.cat([x, x1], dim=1)
        x = self.dropout(x)
        x = self.out_conv(x)  # Shape: [B, num_classes, H, W]
        return x

6. Prepare Datasets and DataLoaders

We split the training data into training and validation sets.

# Prepare datasets
train_dataset = ARCDataset(train_challenges, train_solutions, augment=True)
train_samples, val_samples = train_test_split(train_dataset.samples, test_size=0.2, random_state=42)
train_dataset.samples = train_samples
val_dataset = ARCDataset(samples=val_samples)

# Prepare dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Use batch_size=1 due to varying grid sizes
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

7. Define Loss Function, Optimizer, and Scheduler

We use Focal Loss to handle class imbalance and AdamW optimizer.

# Implement Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=3, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.view(-1))
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()

criterion = FocalLoss(gamma=3)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

8. Training Loop with Validation and Early Stopping

We train the model, validate it, and implement early stopping to prevent overfitting.

# Instantiate the model
model = UNet().to(device)

# Training loop
num_epochs = 64
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 16  # Early stopping patience

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        # Prepare inputs and targets
        inputs = torch.nn.functional.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float().to(device)
        if targets is not None:
            targets = targets.to(device).long()
        else:
            continue  # Skip if targets are None
        optimizer.zero_grad()
        outputs = model(inputs)
        # Resize outputs to match target size
        outputs = nn.functional.interpolate(outputs, size=targets.shape[1:], mode='nearest')
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = torch.nn.functional.one_hot(inputs, num_classes=10).permute(0, 3, 1, 2).float().to(device)
            if targets is not None:
                targets = targets.to(device).long()
            else:
                continue  # Skip if targets are None
            outputs = model(inputs)
            outputs = nn.functional.interpolate(outputs, size=targets.shape[1:], mode='nearest')
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print("Early stopping triggered.")
            break

9. Load the Best Model

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

10. Generate Predictions

We define a function to generate predictions for the test data.

# Function to generate predictions
def generate_predictions(model, challenges, solutions=None):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for task_id, task in challenges.items():
            task_predictions = []
            for idx, test_example in enumerate(task['test']):
                input_grid = np.array(test_example['input'])
                input_tensor = torch.tensor(input_grid, dtype=torch.long)
                # One-hot encode input
                inputs = torch.nn.functional.one_hot(input_tensor, num_classes=10)\
                         .permute(2, 0, 1).unsqueeze(0).float().to(device)
                outputs = model(inputs)
                # Determine the expected output size
                if solutions is not None and task_id in solutions:
                    expected_output_grid = np.array(solutions[task_id][idx])
                    output_size = expected_output_grid.shape
                else:
                    output_size = inputs.shape[2:]
                # Resize outputs to match expected output size
                outputs = nn.functional.interpolate(outputs, size=output_size, mode='nearest')
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                # Duplicate the prediction to get two attempts
                task_predictions.append(pred.tolist())  # attempt_1
                task_predictions.append(pred.tolist())  # attempt_2
            predictions[task_id] = task_predictions
    return predictions

11. Validate Submission Format

Before submitting, we validate the submission file to ensure it meets the competition’s requirements.

# Validation function to check submission format
def validate_submission(submission_path, test_challenges):
    with open(submission_path, 'r') as f:
        submission = json.load(f)
    for task_id in test_challenges.keys():
        if task_id not in submission:
            print(f"Task ID {task_id} is missing in submission.")
            return False
        test_inputs = test_challenges[task_id]['test']
        submissions = submission[task_id]
        if len(submissions) != len(test_inputs):
            print(f"Mismatch in number of test inputs and submissions for task {task_id}.")
            return False
        for idx, sub in enumerate(submissions):
            if 'attempt_1' not in sub or 'attempt_2' not in sub:
                print(f"Missing attempts in submission for task {task_id}, input {idx}.")
                return False
            # Check that the predictions are lists of lists
            if not isinstance(sub['attempt_1'], list) or not isinstance(sub['attempt_2'], list):
                print(f"Invalid prediction format for task {task_id}, input {idx}.")
                return False
    print("Submission file passed validation.")
    return True

12. Prepare Submission File

We prepare the submission.json file with the correct format.

# Generate predictions for test data
test_predictions = generate_predictions(model, test_challenges)

# Prepare submission with both attempt_1 and attempt_2 per test input
submission = {}
for task_id in test_challenges.keys():
    task_submission = []
    num_test_inputs = len(test_challenges[task_id]['test'])
    preds = test_predictions.get(task_id, [])
    # Ensure that preds has exactly 2 * num_test_inputs entries
    if len(preds) < 2 * num_test_inputs:
        preds.extend([preds[-1]] * (2 * num_test_inputs - len(preds)))  # Pad if necessary
    for idx in range(num_test_inputs):
        attempt_1 = preds[idx * 2]
        attempt_2 = preds[idx * 2 + 1]
        task_submission.append({"attempt_1": attempt_1, "attempt_2": attempt_2})
    submission[task_id] = task_submission

# Save submission file
with open('submission.json', 'w') as f:
    json.dump(submission, f)
print("Submission file 'submission.json' created.")

# Validate the submission file
validate_submission('submission.json', test_challenges)

13. Evaluate Model Performance

We evaluate the model on the validation and training sets to check its performance.

# Function to score predictions
def score_predictions(predictions, solutions):
    total_elements = 0
    correct_elements = 0
    for task_id, task_solutions in solutions.items():
        preds = predictions.get(task_id, [])
        for idx, solution in enumerate(task_solutions):
            # Since we have two predictions per test input, get the first one for scoring
            pred = preds[idx*2]  # idx*2 because predictions are duplicated
            pred_array = np.array(pred)
            solution_array = np.array(solution)
            correct = np.sum(pred_array == solution_array)
            total = pred_array.size
            correct_elements += correct
            total_elements += total
    final_score = (correct_elements / total_elements) * 100 if total_elements > 0 else 0
    return final_score

# Generate and evaluate predictions on validation data
val_challenges = {'val_task': {'test': [{'input': s['input'].tolist()} for s in val_dataset.samples]}}
val_solutions = {'val_task': [s['output'].tolist() for s in val_dataset.samples]}
val_predictions = generate_predictions(model, val_challenges, solutions=val_solutions)
val_score = score_predictions(val_predictions, val_solutions)
print(f"Validation Score: {val_score:.2f}%")

# Generate and evaluate predictions on a subset of training data
train_challenges_subset = {'train_task': {'test': [{'input': s['input'].tolist()} for s in train_dataset.samples[:100]]}}
train_solutions_subset = {'train_task': [s['output'].tolist() for s in train_dataset.samples[:100]]}
train_predictions = generate_predictions(model, train_challenges_subset, solutions=train_solutions_subset)
train_score = score_predictions(train_predictions, train_solutions_subset)
print(f"Training Score: {train_score:.2f}%")

14. Visualize Some Predictions

We visualize some predictions from the validation set to see how the model is performing.

# Visualize a few validation predictions
def visualize_predictions(predictions, challenges, solutions, num_examples=3):
    preds = predictions['val_task']
    inputs = [np.array(ex['input']) for ex in challenges['val_task']['test']]
    outputs = [np.array(out) for out in solutions['val_task']]
    for idx in range(num_examples):
        input_grid = inputs[idx]
        output_grid = outputs[idx]
        prediction_grid = np.array(preds[idx*2])  # idx*2 because of duplication
        title = f'Validation Example {idx}'
        plot_task(input_grid, output_grid, prediction_grid, title)

# Call the visualization function
visualize_predictions(val_predictions, val_challenges, val_solutions)

