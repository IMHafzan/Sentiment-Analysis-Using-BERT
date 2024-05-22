from src.train_model2 import train_model, load_data

def main():
    train_data, val_data = load_data()
    train_model(train_data, val_data)

if __name__ == "__main__":
    main()
