import json
import os

def save_state_to_json(state, filepath):
    """Saves a dictionary state to a JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            state_to_save = state.copy()
            if 'sample_stats' in state_to_save:
                # This part is complex and can be slow for large datasets.
                # For simplicity, we save it as is.
                pass 
            json.dump(state_to_save, f, indent=4)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving state to {filepath}: {e}")
        return False

def load_state_from_json(filepath):
    """Loads a dictionary state from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            state = json.load(f)
            # JSON saves dict keys as strings, so convert sample_stats keys back to int
            if 'sample_stats' in state:
                state['sample_stats'] = {int(k): v for k, v in state['sample_stats'].items()}
            return state
    except (IOError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading state from {filepath}: {e}")
        return None