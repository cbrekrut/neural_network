import numpy as np

# Определение сигмоидной функции активации
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Определение производной сигмоидной функции
def sigmoid_derivative(x):
    return x * (1 - x)

# Входные данные для обучения
training_input = np.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]])

# Ожидаемые выходные данные для обучения
training_output = np.array([[0, 1, 1, 0]]).T

# Задание случайного семени для воспроизводимости результатов
np.random.seed(1)

# Инициализация синаптических весов случайными значениями
synaptic_weights = 2 * np.random.random((3, 1)) - 1

# Цикл обучения
for i in range(30000):
    # Прямое распространение
    input_layer = training_input
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # Вычисление ошибки и корректировка синаптических весов
    error = training_output - outputs
    adjustments = error * sigmoid_derivative(outputs)
    synaptic_weights += np.dot(input_layer.T, adjustments)

# Вывод результатов
print("Результаты после обучения:")
print(outputs)

new_input = np.array([1,1,0])
out = sigmoid(np.dot(new_input,synaptic_weights))
print("новая ситуация:")
print(round(float(out)))