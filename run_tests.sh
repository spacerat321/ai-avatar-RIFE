#!/bin/bash

# 1. Use a default venv folder if it exists
VENV_PATH="./venv"

# 2. Check if virtual environment exists
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment from $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "No virtual environment found at $VENV_PATH. Using system Python."
fi

# 3. Fix for ~/.local/bin if pytest isn't globally linked
export PATH="$HOME/.local/bin:$PATH"

# 4. Run pytest with verbose output
echo "Running tests with pytest..."
python3 -m pytest -v tests/

# 5. Optionally deactivate venv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi
