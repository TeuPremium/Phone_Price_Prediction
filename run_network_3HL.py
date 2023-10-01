import pandas as pd
import numpy as np
from Neural_Network import Layer_Dense, activation_ReLU, Activation_Softmax_Loss_CategoricalCrossentropy, Optimizer_SGD, activation_softmax
import matplotlib.pyplot as plt
import seaborn as sns

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

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1
    return cm

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


       # Initializing the network and defining stricture:
    n_inputs = len(X_train_normalized[0])
    n_outputs = 4  # 4 -> output classes based on the provided dataset
   
    dense1 = Layer_Dense(n_inputs, 32)  # usando uma camada inicial com 32 neurônios
    activation1 = activation_ReLU()
    dense2 = Layer_Dense(32, 16)  # primeira camada escondida com 16 neurônios e ReLU
    activation2 = activation_ReLU()  # ativação ReLU para a primeira camada escondida
    dense3 = Layer_Dense(16, 8)  # segunda camada escondida com 8 neurônios e ReLU
    activation3 = activation_ReLU()  # ativação ReLU para a segunda camada escondida
    dense4 = Layer_Dense(8, n_outputs)  # Camada de saída
    activation4 = activation_softmax()


    softmax_loss = Activation_Softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD(learning_rate=0.1)  # Adjust the learning rate as needed

    # setting hyperparameters
    num_epochs = 1500
    rms_errors = [] 


    train_losses = []
    test1_losses = []

    
    for epoch in range(num_epochs):
        # Forward pass
        dense1_output = dense1.forward(X_train_normalized)
        activation1_output = activation1.forward(dense1_output)
        dense2_output = dense2.forward(activation1_output)
        activation2_output = activation2.forward(dense2_output)
        dense3_output = dense3.forward(activation2_output)
        activation3_output = activation3.forward(dense3_output)
        dense4_output = dense4.forward(activation3_output)
        train_output = activation4.forward(dense4_output)

        # Calcular a perda
        loss = softmax_loss.forward(train_output, y_train)

        # Backward pass
        y_one_hot = np.eye(n_outputs)[y_train]

        grad_loss = (train_output - y_one_hot) / len(y_train)
        softmax_loss.backward(grad_loss, y_train)
        dense4.backward(grad_loss)
        activation3.backward(dense4.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Atualizar pesos e biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.update_params(dense4)

        # calculate RMS Error
        squared_errors = (train_output - y_one_hot)**2
        mean_squared_error = np.mean(squared_errors)
        rms_error = np.sqrt(mean_squared_error)
        rms_errors.append(rms_error)  # Adicionando à lista de erros RMS

        # Imprimir o progresso do treinamento
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, RMS Error: {rms_error:.4f}")

            # test1 pass
        dense1_output = dense1.forward(X_val_normalized)
        activation1_output = activation1.forward(dense1_output)
        dense2_output = dense2.forward(activation1_output)
        activation2_output = activation2.forward(dense2_output)
        dense3_output = dense3.forward(activation2_output)
        activation3_output = activation3.forward(dense3_output)
        dense4_output = dense4.forward(activation3_output)
        test1_output = activation4.forward(dense4_output)

        test1_loss = softmax_loss.forward(test1_output, y_val)


        # Saves the losses to a list
        test1_losses.append(test1_loss)
        train_losses.append(loss)


     

     # Obtemos a classe predita para cada exemplo
    train_predicted_classes = np.argmax(train_output, axis=1)
    # Calculamos a acurácia
    accuracy = calculate_accuracy(y_train, train_predicted_classes)
    print(f"Train Accuracy: {accuracy*100:.2f}%")




    test1_predicted_classes = np.argmax(test1_output, axis=1)
    accuracy = calculate_accuracy(y_val, test1_predicted_classes)
    print(f"test1 Accuracy: {accuracy*100:.2f}%")

        # Test pass
    dense1_output = dense1.forward(X_test_normalized)
    activation1_output = activation1.forward(dense1_output)
    dense2_output = dense2.forward(activation1_output)
    activation2_output = activation2.forward(dense2_output)
    dense3_output = dense3.forward(activation2_output)
    activation3_output = activation3.forward(dense3_output)
    dense4_output = dense4.forward(activation3_output)
    test_output = activation4.forward(dense4_output)

    test_loss = softmax_loss.forward(test_output, y_test)


        # Calculating accuracy 
    test_predicted_classes = np.argmax(test_output, axis=1)
    accuracy = calculate_accuracy(y_test, test_predicted_classes)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    confusion_mat_test1 = confusion_matrix(y_val, test1_predicted_classes, num_classes=4)
    print("Confusion Matrix for test1:")
    print(confusion_mat_test1)

    confusion_mat_test = confusion_matrix(y_test, test_predicted_classes, num_classes=4)
    print("\nConfusion Matrix for Test:")
    print(confusion_mat_test)

    # Plotting RMS error per epoch
    plt.figure(figsize=(5, 5))
    plt.plot(range(epoch+1), rms_errors, label='Train RMS per epoch', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Train RMS Error')
    plt.title(f'RMS Error per epoch {epoch+1}')
    plt.legend()
    plt.show()

    # Plotting loss per epoch
    plt.figure(figsize=(5, 5))
    plt.plot(range(epoch+1), train_losses, label='test1 Loss por Época', color='orange')
    plt.plot(range(epoch+1), train_losses, label='Train Loss por Época', color='blue')
    plt.xlabel('Época')
    plt.ylabel('Loss (-log)')
    plt.title('Loss per Époch')
    plt.legend()
    plt.show()


    #  <--------------------------### PLOT THE GRAPHS ###---------------------------->
    if isinstance(X_train_normalized, pd.DataFrame):
        X_train_features = X_train_normalized.iloc[:, :20]
    else:
        X_train_features = X_train_normalized[:, :20]

    # Assuming y_train is a 1D numpy array
    feature_names = ["battery_power", "blue", "clock_speed", "dual_sim", 
                    "fc", "four_g", "int_memory", "m_dep", "mobile_wt", 
                    "n_cores", "pc", "px_height", "px_width", "ram", 
                    "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"]

    plt.figure(figsize=(12, 60))
    for i, feature in enumerate(feature_names):
        plt.subplot(11, 2, i+1)
        if isinstance(X_train_normalized, pd.DataFrame):
            plt.boxplot(X_train_features[feature])
        else:
            plt.boxplot(X_train_features[:, i])
        plt.xlabel(feature)
        plt.title(f'{feature} Distribution')

    plt.tight_layout()
    plt.show()


    # Assuming X_train_normalized is a DataFrame or a 2D numpy array
    # If it's a DataFrame, select the "clock_speed" column for plotting
    if isinstance(X_train_normalized, pd.DataFrame):
        clock_speed_data = X_train_normalized["clock_speed"]
    else:
        clock_speed_data = X_train_normalized[:, 2]  # Assuming clock_speed is the third feature

    # Create a DataFrame for easier plotting
    data = pd.DataFrame({"clock_speed": clock_speed_data, "price_range": y_train})

    # Create a boxplot for clock_speed by price_range
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="price_range", y="clock_speed", data=data)
    plt.xlabel('Price Range')
    plt.ylabel('Clock Speed')
    plt.title('Clock Speed Distribution by Price Range')
    plt.show()

        # Assuming X_train_normalized is a DataFrame or a 2D numpy array
    # If it's a DataFrame, select the "fc" column for plotting
    if isinstance(X_train_normalized, pd.DataFrame):
        fc_data = X_train_normalized["fc"]
    else:
        fc_data = X_train_normalized[:, 4]  # Assuming fc is the fifth feature

    # Create a DataFrame for easier plotting
    data = pd.DataFrame({"Front Camera (fc)": fc_data, "Price Range": y_train})

    # Create a boxplot for fc by price_range
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Price Range", y="Front Camera (fc)", data=data)
    plt.xlabel('Price Range')
    plt.ylabel('Front Camera (fc)')
    plt.title('Front Camera (fc) Distribution by Price Range')
    plt.show()

    # Assuming X_train_normalized is a DataFrame or a 2D numpy array
    # If it's a DataFrame, select the "four_g" column for plotting
    if isinstance(X_train_normalized, pd.DataFrame):
        four_g_data = X_train_normalized["four_g"]
    else:
        four_g_data = X_train_normalized[:, 5]  # Assuming four_g is the sixth feature

    # Create a DataFrame for easier plotting
    data = pd.DataFrame({"Four G (four_g)": four_g_data, "Price Range": y_train})

    # Create a boxplot for four_g by price_range
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Price Range", y="Four G (four_g)", data=data)
    plt.xlabel('Price Range')
    plt.ylabel('Four G (four_g)')
    plt.title('Four G (four_g) Distribution by Price Range')
    plt.show()




if __name__ == "__main__":
    main()


