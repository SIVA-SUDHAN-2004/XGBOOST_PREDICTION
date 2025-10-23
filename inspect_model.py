import pickle

with open("productivity_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model type:", type(model))

# Try to print the expected feature names
if hasattr(model, "feature_names_in_"):
    print("Expected features:")
    print(model.feature_names_in_)
else:
    print("Model doesn't have feature_names_in_")
