from tensorflow import keras, optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import layers
from keras import regularizers


(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# Plots a single digit from the data
sns.heatmap(train_data[1, :, :])
plt.show()

# Reshapes the data to work in a FFN

train_data = train_data.reshape((60000, 28*28))
test_data = test_data.reshape((10000, 28*28))

num_classes =10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

lambda_vals = [0.1, 0.001, 0.00001]
# L2
kernel_regularizes_L2 = [regularizers.l2(lambda_vals[0]), regularizers.l2(lambda_vals[1]), regularizers.l2(lambda_vals[2])]
# L1
kernel_regularizes_L1 = [regularizers.l1(lambda_vals[0]), regularizers.l1(lambda_vals[1]), regularizers.l1(lambda_vals[2])]




results_reg_test = { 0.1 : [], 0.001 : [], 0.00001 : []}

def create_base_model():
	model = create_model(activation='relu', number_of_dense_layers=4,  regularizer=None, input_node_number=300, dense_node_number=128, dropout=False)
	model_history = fit_model(model)
	return model_history

def test(model):
	history_dict_model = model.history
	accuracy = history_dict_model['accuracy']
	loss = history_dict_model['loss']
	val_accuracy = history_dict_model['val_accuracy']
	val_loss = history_dict_model['val_loss']
	epochs = range(1, len(val_loss) + 1)

	plt.plot(epochs, loss, 'g', label="Training loss")
	plt.plot(epochs, val_loss, 'b', label="Validation loss")

	plt.plot(epochs, accuracy, 'g', label="Training accuracy")
	plt.plot(epochs, val_accuracy, 'r', label="Validation accuracy")

	plt.title("Training and validation loss within regularization")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")

	plt.legend()
	plt.show()



def create_model(activation, number_of_dense_layers,  regularizer, input_node_number, dense_node_number, dropout):
	model = keras.models.Sequential()
	model.add(Dense(input_node_number, activation=activation, input_shape=[28*28, ], kernel_regularizer=regularizer))
	for layer in range(number_of_dense_layers):
		model.add(Dense(dense_node_number, activation=activation, kernel_regularizer=regularizer))
		if dropout:
			model.add(Dropout(0.5))

	model.add(Dense(num_classes, activation="softmax"))

	model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def fit_model(model):
	model_history = model.fit(
		train_data, train_labels,
		epochs=5,
		batch_size=32,
		validation_data=(test_data, test_labels))
	return model_history



def test_model(regul_arr, input_node_number, dense_node_number, dropout, number_of_dense_layers, activation_for_hidden_layers, tested_data_set, examined_data):
	iter = 0
	if regul_arr is not None:
		for regul in regul_arr:
			model = create_model(activation=activation_for_hidden_layers, number_of_dense_layers=number_of_dense_layers, regularizer=regul,  input_node_number=input_node_number, dense_node_number=dense_node_number, dropout=dropout)
			model_history = fit_model(model)
			history_dict_model = model_history.history
			tested_data_set.get(examined_data[iter]).append(history_dict_model)
			iter += 1
	else:
		model = create_model(activation=activation_for_hidden_layers, number_of_dense_layers=number_of_dense_layers,
		                     regularizer=None, input_node_number=input_node_number,
		                     dense_node_number=dense_node_number, dropout=dropout)
		model_history = fit_model(model)
		history_dict_model = model_history.history
		tested_data_set.get(examined_data[iter]).append(history_dict_model)


		"""
		accuracy = history_dict_model['accuracy']
		loss = history_dict_model['loss']
		val_accuracy = history_dict_model['val_accuracy']
		val_loss = history_dict_model['val_loss']
		epochs = range(1, len(val_loss) + 1)

		plt.plot(epochs, loss, 'g', label="Training loss")
		plt.plot(epochs, val_loss, 'b', label="Validation loss")

		plt.plot(epochs, accuracy, 'g', label="Training accuracy")
		plt.plot(epochs, val_accuracy, 'r', label="Validation accuracy")

		plt.title("Training and validation loss within regularization")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")

		plt.legend()
		plt.show()
		"""

def compare_results(dict, whatWeCompare):

	for key, values in dict.items():
		iter=0
		for val in values:
			plt.subplot(1, len(values), iter+1)  # row 1, col 2 index 1
			accuracyL2 = val['accuracy']
			lossL2 = val['loss']
			val_accuracyL2 = val['val_accuracy']
			val_lossL2 = val['val_loss']

			epochs = range(1, len(val_lossL2) + 1)

			iter += 1
			plt.subplot(1, 2, 1)
			plt.plot(epochs, lossL2, 'g', label=f"Training loss for {val} ")
			plt.plot(epochs, val_lossL2, 'b', label=f"Validation loss for {val}  ")

			plt.title(f"Training and validation loss comparision with a parameter - {key} for {whatWeCompare} ")
			plt.xlabel("Epochs")
			plt.ylabel("Loss")
			plt.legend()

			plt.subplot(1, 2, 2)
			plt.plot(epochs, accuracyL2, 'g', label=f"Training accuracy for {val}  ")
			plt.plot(epochs, val_accuracyL2, 'r', label=f"Validation accuracy  for {val} ")

			plt.title(f"Training and validation accuracy comparision with a parameter - {key} for {whatWeCompare} ")
			plt.xlabel("Epochs")
			plt.ylabel("Loss")
			plt.legend()
	plt.show()


def compare_results_2(dict, whatWeCompare):

	for key, values in dict.items():
		iter = 0
		for val in values:
			plt.subplot(1, len(values), iter + 1)  # row 1, col 2 index 1
			accuracyL2 = val['accuracy']
			lossL2 = val['loss']
			val_accuracyL2 = val['val_accuracy']
			val_lossL2 = val['val_loss']

			epochs = range(1, len(val_lossL2) + 1)

			iter += 1
			plt.subplot(1, 2, 1)
			plt.plot(epochs, lossL2, 'g', label=f"Training loss  ")
			plt.plot(epochs, val_lossL2, 'b', label=f"Validation loss")
			plt.title(f"Training and validation loss  ")
			plt.xlabel("Epochs")
			plt.ylabel("Loss")
			plt.legend()


			plt.subplot(1, 2, 2)
			plt.plot(epochs, accuracyL2, 'g', label=f"Training accuracy for  ")
			plt.plot(epochs, val_accuracyL2, 'r', label=f"Validation accuracy  for ")
			plt.title(f"Training and validation loss comparision")
			plt.xlabel("Epochs")
			plt.ylabel("Loss")
			plt.legend()
	plt.show()


def test_reg():
	tested_data_set = {0.1 : [], 0.001 : [], 0.00001 : []}
	test_model(kernel_regularizes_L2, number_of_dense_layers=4, input_node_number=32, dense_node_number=32, dropout=False, activation_for_hidden_layers='relu', tested_data_set=tested_data_set, examined_data=lambda_vals)
	test_model(kernel_regularizes_L1, number_of_dense_layers=4, input_node_number=32, dense_node_number=32, dropout=False, activation_for_hidden_layers='relu', tested_data_set=tested_data_set, examined_data=lambda_vals)
	compare_results(dict=tested_data_set, whatWeCompare="L2/L1")



def test_with_dropout():
	tested_data_set = {0.1: [], 0.001: [], 0.00001: []}
	test_model(kernel_regularizes_L2, number_of_dense_layers=4, input_node_number=32, dense_node_number=32, dropout=True, activation_for_hidden_layers='relu' ,tested_data_set=tested_data_set, examined_data=lambda_vals)
	test_model(kernel_regularizes_L2, number_of_dense_layers=4, input_node_number=32, dense_node_number=32, dropout=False, activation_for_hidden_layers='relu', tested_data_set=tested_data_set, examined_data=lambda_vals)
	compare_results(tested_data_set, whatWeCompare= "with dropout/ without dropout")

"units, the amount of layers, activation functions, etc.to obtain the best accuracy you can."

def test_number_of_nodes():
	power_of_two = [2, 3, 4, 5, 6]
	tested_data_set = {2: [], 3: [], 4: [], 5: [], 6: []}
	for power in power_of_two:
		test_model(None, number_of_dense_layers=4, input_node_number=pow(2, power), dense_node_number=pow(2, power), dropout=True, activation_for_hidden_layers='relu',tested_data_set=tested_data_set, examined_data=power_of_two)
		compare_results_2(tested_data_set, f"number of nodes- 2^ {power}")


def test_number_of_layers():
	power_of_two = [1, 2, 3, 4, 5]
	tested_data_set = {1: [], 2: [], 3: [], 4: [], 5: []}
	for power in power_of_two:
		test_model(None, number_of_dense_layers=4, input_node_number=pow(2, power), dense_node_number=pow(2, power), dropout=True, activation_for_hidden_layers='relu',tested_data_set=tested_data_set, examined_data=power_of_two)
		compare_results_2(tested_data_set, f"number of layers - 2^{power}")


def test_AF():
	afs = ['selu', 'relu', 'sigmoid', 'tanh' ]
	tested_data_set = {'selu': [], 'relu': [], 'sigmoid': [], 'tanh': []}
	for af in afs:
		test_model(None, number_of_dense_layers=4, input_node_number=32, dense_node_number=32, dropout=True, activation_for_hidden_layers=af,tested_data_set=tested_data_set, examined_data=afs)
		compare_results_2(tested_data_set, f"activation functions - {af}")



def save_model_to_file(model, filename):
	model.save(filename)

def load_model_from_file(filename):
	return load_model(filename)

def create_early_stopping_model(model):
	es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
	model_history = model.fit(train_data, train_labels, callbacks=[es_callback])
	return model_history


def compare_model_performance(models):
	iter = 0
	for model in models:
		plt.subplot(1, len(models), iter + 1)  # row 1, col 2 index 1
		test(model)
		iter += 1


def test_early_stopping():
	base_model = create_model(activation='relu', number_of_dense_layers=4,  regularizer=None, input_node_number=300, dense_node_number=128, dropout=False)
	models = [create_early_stopping_model(base_model), create_base_model()]
	compare_model_performance(models)


test_number_of_layers()
test_AF()
test_early_stopping()



#przejdz przez historie porównaj wszystkie lossy
#sprawdz overfitting if overfitting to nie bierz jak nie to bierz
#i wez te parametry gdzie loss najmniejszy

#zmien kolor na wkyresach

#dopisz funkcje ddo porownywania na jednym wykresie wszstkich wartosci dla roznych nodow i roznych layerow i roznych aktyw funckcji


