import torch
import yaml
import copy
import os
import argparse
from pathlib import Path
import time
import shutil
import multiprocessing

# macOS stability fix: Avoid EXC_BAD_ACCESS in fork() after framework init
if hasattr(time, "tzset") and os.name == "posix":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Minimal model for testing
class TestModel(torch.nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_device(cfg):
    pref = cfg.get('device', 'cpu')
    if pref == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if pref == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def run_hardware_test(config_path, master_model_path=None):
    print(f"--- Running Hardware Test with config: {config_path} ---")
    cfg = load_config(config_path)
    device = get_device(cfg)
    print(f"Target Device: {cfg['device']}, Actual Device: {device}")
    
    spending_limit = cfg.get('spending_limit_usd', 100.0)
    print(f"Spending Limit: ${spending_limit}")

    # 1. Handle "ENSURE ML MODEL COPIES RECIEVE TRAINING NOT ORIGIONAL"
    # If a master model path is provided, we copy it first.
    if master_model_path:
        master_model_path = Path(master_model_path)
        print(f"Using existing master model: {master_model_path}")
        
        # Create a "copy" for training
        training_dir = Path("checkpoints/training_copies")
        training_dir.mkdir(parents=True, exist_ok=True)
        training_model_path = training_dir / f"training_copy_{master_model_path.name}"
        
        print(f"Cloning master model to training copy: {training_model_path}")
        shutil.copy2(master_model_path, training_model_path)
        
        # Load the training copy
        # In a real scenario: model = load_model(training_model_path)
        master_model = TestModel(128, 256, 10).to(device)
        # Simulate loading weights
        # master_model.load_state_dict(torch.load(training_model_path))
    else:
        # Create a fresh master model in memory for this benchmark
        master_model = TestModel(128, 256, 10).to(device)
        print("Fresh master model initialized in memory.")

    master_weights_initial = copy.deepcopy(master_model.state_dict())

    # Create the training model (the "copy" that will actually be trained)
    training_model = copy.deepcopy(master_model).to(device)
    print("Created in-memory training copy of the model.")

    # Verify cloning worked
    for (name1, p1), (name2, p2) in zip(master_model.named_parameters(), training_model.named_parameters()):
        if not torch.equal(p1, p2):
            raise RuntimeError(f"Cloning failed for parameter {name1}")
    print("Verification: Master and training copy have identical initial weights.")

    # 3. Setup Optimizer for training copy
    optimizer = torch.optim.Adam(training_model.parameters(), lr=float(cfg['optim']['lr']))
    
    # Mock data for testing
    batch_size = cfg['dataloader']['batch_size']
    mock_input = torch.randn(batch_size, 128).to(device)
    mock_target = torch.randint(0, 10, (batch_size,)).to(device)

    # 4. Training loop on COPY
    print(f"Starting training loop for {cfg['training']['max_steps']} steps...")
    training_model.train()
    master_model.eval() # Ensure master is in eval mode

    start_time = time.time()
    for step in range(cfg['training']['max_steps']):
        # Simple simulated cost check: $0.05 per step
        current_cost = step * 0.05 
        if current_cost > spending_limit:
            print(f"STOPPING: Spending limit of ${spending_limit} reached at step {step}! (Cost: ${current_cost:.2f})")
            break
            
        optimizer.zero_grad()
        output = training_model(mock_input)
        loss = torch.nn.functional.cross_entropy(output, mock_target)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step}/{cfg['training']['max_steps']} - Loss: {loss.item():.4f} - Simulated Cost: ${current_cost:.2f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

    # 5. Verify integrity: Master MUST be unchanged, Training MUST be changed
    print("\n--- Integrity Verification ---")
    
    master_unchanged = True
    for name, p in master_model.named_parameters():
        if not torch.equal(p.data, master_weights_initial[name]):
            print(f"CRITICAL ERROR: Master model parameter {name} was modified!")
            master_unchanged = False
            
    training_changed = False
    for name, p in training_model.named_parameters():
        if not torch.equal(p.data, master_weights_initial[name]):
            training_changed = True
            break
            
    if master_unchanged:
        print("SUCCESS: Original master model remains UNCHANGED.")
    else:
        print("FAILURE: Original master model was CORRUPTED during training.")
        
    if training_changed:
        print("SUCCESS: Training copy has been UPDATED.")
    else:
        print("WARNING: Training copy weights did not change (loss might be zero or lr too low).")

    print(f"--- Hardware Test {config_path} Finished ---\n")
    return master_unchanged and training_changed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--master", type=str, help="Optional path to master model file")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
    else:
        run_hardware_test(args.config, args.master)
