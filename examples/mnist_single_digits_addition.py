import torch
import ltn


def main():
    data = ltn.utils.get_mnist_dataset_for_digits_addition(single_digit=True)
    print(data[0])


if __name__ == "__main__":
    main()