from pre_processing_A import get_data
from A1.A1 import model_a1

def main():

    # get data for task A1
    Y_train_a1, Y_test_a1, X_test_a1, X_train_a1 = get_data('gender')
    # get data for task A2 
    Y_train_a2, Y_test_a2, X_test_a2, X_train_a2 = get_data('smiling')

    model_a1(Y_train_a1, Y_test_a1, X_test_a1, X_train_a1)


if __name__ == "__main__":
    main()