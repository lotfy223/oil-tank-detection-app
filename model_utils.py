import os
import streamlit as st

@st.cache_resource
def load_detection_model():
    """
    Loads the Keras CNN model for oil tank detection.
    Supports both TensorFlow and PyTorch backend via Keras 3 depending on availability.
    Returns:
        model: Loaded Keras model.
        backend: Str name of the active backend ('tensorflow' or 'torch').
    """
    # 1. Determine and configure the backend
    try:
        import tensorflow as tf
        backend = "tensorflow"
    except ImportError:
        # Fallback to PyTorch backend for Keras if TensorFlow is not available
        os.environ["KERAS_BACKEND"] = "torch"
        backend = "torch"
        
    import keras
    
    # 2. Search for the Keras model file
    model_paths = [
        "oil_tank_detection_model.keras",
        "oil_tank_detection_model(1).keras"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    if model_path is None:
        # Fallback: find any .keras file in the current directory
        try:
            keras_files = [f for f in os.listdir(".") if f.endswith(".keras")]
            if keras_files:
                model_path = keras_files[0]
        except Exception:
            pass
            
    if model_path is None:
        raise FileNotFoundError(
            "Could not find the Keras model file in the current directory. "
            "Please ensure 'oil_tank_detection_model.keras' or 'oil_tank_detection_model(1).keras' is present."
        )
        
    # 3. Load the model using Keras
    try:
        model = keras.models.load_model(model_path)
        return model, backend
    except Exception as e:
        raise RuntimeError(
            f"Error loading model from '{model_path}' using backend '{backend}': {str(e)}"
        )
