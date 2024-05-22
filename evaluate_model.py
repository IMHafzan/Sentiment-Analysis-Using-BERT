from src.evaluate_model_2 import evaluate_model, load_data

def main():
    val_data = load_data()
    evaluate_model(val_data)

if __name__ == "__main__":
    main()
