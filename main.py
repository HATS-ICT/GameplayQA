from datasets import load_dataset

def main():
    print("Hello from gameplayqa!")


if __name__ == "__main__":
    ds = load_dataset("wangyz1999/GameplayQA")
    print(ds)
