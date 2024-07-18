import pickle
from ml_logic.model import initialize_model, compile_model

# Create model.
# model = ml_logic.model.initialize_model()
# model = ml_logic.model.compile_model(model)

# Load the model at startup and do not wait for the request "predict".
# Shall be run only once in the lifetime of the programme.
def load_model():
    model = False
    try:
        model = pickle.load(open("model/my_model.pkl", 'rb'))
    except Exception:
        print("Model could not be loaded...")
    if not model:
        model = initialize_model()
        model = compile_model(model)
        # (tbd): Save the model.
        try:
            with open("model/my_model.pkl", "wb") as file:
                pickle.dump(model, file)
            print("Saved new model.")
        except Exception:
            pass
    return model


if __name__ == "__main__":
    # Create the dummy model.

    # Apply the model.
    ml_logic.model.model_summary(model)
    ml_logic.model.model_test(model)
