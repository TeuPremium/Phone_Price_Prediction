import pandas as pd
import numpy as np
from Neural_Network import Layer_Dense, activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD, activation_softmax


class import_datasets:
    def __init__(self):
        self.train = 0
        self.test = 0
        self.validate = 0
    
    def load_train_data(self):
        dataset = pd.read_csv('train.csv')
        self.train_data = dataset
        return dataset

    def load_test_data(self):
        dataset = pd.read_csv('test.csv')
        # self.load_test_data = dataset
        return dataset

    def custom_train_val_test_split(self, X, y, val_size, test_size, random_state):
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(len(X))
        
        val_set_size = int(len(X) * val_size)
        test_set_size = int(len(X) * test_size)
        
        val_indices = shuffled_indices[:val_set_size]
        test_indices = shuffled_indices[val_set_size : val_set_size + test_set_size]
        train_indices = shuffled_indices[val_set_size + test_set_size:]
        
        X_train, X_val, X_test = X.iloc[train_indices], X.iloc[val_indices], X.iloc[test_indices]
        y_train, y_val, y_test = y.iloc[train_indices], y.iloc[val_indices], y.iloc[test_indices]

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        X_train = pd.DataFrame(X_train)

        return  X_train.values.tolist(), X_val.values.tolist(), X_test.values.tolist(),\
                y_train.values.tolist(), y_val.values.tolist(), y_test.values.tolist()




# Copying data and labels

print("starting the training:")

def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized

def calculate_accuracy(y_true, y_pred):
    # Assuming y_true and y_pred are numpy arrays
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def main():
    val_size = 0.1
    test_size = 0.1
    random_state = 133

    p = import_datasets()
    train_data = p.load_train_data()
    # test_data = p.load_test_data()
    X = train_data.drop('price_range', axis=1)
    y = train_data['price_range']

    X_train, X_val, X_test, y_train, y_val, y_test = p.custom_train_val_test_split(X, y, val_size, test_size, random_state)

    X_train, X_val, X_test, y_train, y_val, y_test =  np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)

    X_train_normalized = normalize_features(X_train)
    X_val_normalized = normalize_features(X_val)
    X_test_normalized = normalize_features(X_test)

        # print(X_train_normalized[0])
        # print(np.shape(X_test_normalized))
        # print(type(X_train_normalized))

       # Initializing the network and defining stricture:
    n_inputs = len(X_train_normalized[0])
    n_outputs = 4  # 4 -> output classes based on the provided dataset
   
    dense1 = Layer_Dense(n_inputs, 24)  # usando uma camada inicial com 24 neurônios
    activation1 = activation_ReLU()
    dense2 = Layer_Dense(24, 16)  # nova camada escondida com 16 neurônios e ReLU
    activation2 = activation_ReLU()  # ativação ReLU para a nova camada escondida
    dense3 = Layer_Dense(16, n_outputs)  # Camada de saída
    activation3 = activation_softmax()


    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=0.05)  # Adjust the learning rate as needed

    # setting hyperparameters
    num_epochs = 5000

    # Training loop
    for epoch in range(num_epochs):
        # Forward pass
        dense1_output = dense1.forward(X_train_normalized)
        activation1_output = activation1.forward(dense1_output)
        dense2_output = dense2.forward(activation1_output)
        activation2_output = activation2.forward(dense2_output)
        dense3_output = dense3.forward(activation2_output)
        output = activation3.forward(dense3_output)

        # Calcular a perda
        loss = softmax_loss.forward(output, y_train)

        # Backward pass
        y_one_hot = np.eye(n_outputs)[y_train]

        grad_loss = (output - y_one_hot) / len(y_train)
        softmax_loss.backward(grad_loss, y_train)
        dense3.backward(grad_loss)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # uodate weights & biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)

        # Print training progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
         # Obtemos a classe predita para cada exemplo

    predicted_classes = np.argmax(output, axis=1)

    # Calculamos a acurácia
    accuracy = calculate_accuracy(y_train, predicted_classes)
    print(f"Accuracy: {accuracy*100:.2f}%")



    # plotting the data

    # # Create a mesh grid
    # x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    # y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
    #                     np.arange(y_min, y_max, 0.1))

    # # Make predictions on the mesh grid
    # mesh_data = np.c_[xx.ravel(), yy.ravel()]
    # mesh_output = model.predict(mesh_data)
    # mesh_output = np.argmax(mesh_output, axis=1)
    # mesh_output = mesh_output.reshape(xx.shape)

    # # Plot the decision boundaries
    # plt.contourf(xx, yy, mesh_output, cmap=plt.cm.RdYlBu, alpha=0.8)

    # # Plot the training points
    # scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis',
    #             edgecolor='k', s=20)
    # legend1 = plt.legend(*scatter.legend_elements(),
    #                     loc="upper right", title="Classes")
    # plt.gca().add_artist(legend1)

    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())

    # plt.show()




if __name__ == "__main__":
    main()

