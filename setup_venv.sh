#!/bin/bash
# Setup script to create virtual environment in /common/users/rm1838/venv

VENV_DIR="/common/users/rm1838/venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Setting up virtual environment in $VENV_DIR..."

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing virtual environment. Exiting."
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements if they exist
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    # Install torch separately from PyTorch index (for CPU version)
    if grep -q "torch==" "$SCRIPT_DIR/requirements.txt"; then
        echo "Installing PyTorch from PyTorch index..."
        pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
        # Remove torch from requirements.txt temporarily to avoid conflicts
        grep -v "^torch==" "$SCRIPT_DIR/requirements.txt" > /tmp/requirements_no_torch.txt
        pip install -r /tmp/requirements_no_torch.txt
        rm /tmp/requirements_no_torch.txt
    else
        pip install -r "$SCRIPT_DIR/requirements.txt"
    fi
else
    echo "Warning: requirements.txt not found. Installing basic packages..."
    pip install numpy pandas matplotlib jupyter ipykernel
    pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Or add this to your ~/.bashrc or ~/.bash_profile:"
echo "  alias activate_venv='source $VENV_DIR/bin/activate'"

