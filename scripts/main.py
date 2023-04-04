from models import linear_regression, svm, bert


def main():
    while True:
        print("Choose model (lr, svm, bert):", end=" ")
        model_choice = str(input())
        if model_choice in ["lr", "svm", "bert"]:
            break
        else:
            print("Please provide a valid model!")

    if model_choice == "lr":
        linear_regression()
    elif model_choice == "svm":
        svm()
    else:
        bert()

    exit(0)


if __name__ == "__main__":
    main()
