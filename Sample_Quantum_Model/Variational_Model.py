import pennylane as qml
from pennylane import numpy as np
import tensorflow as tf
import driving_data

# Load the training data
batch_size = 100
X_train, y_train = driving_data.LoadTrainBatch(batch_size)

# Normalize the input data
# Convert the list of arrays to a NumPy array
X_train = np.array(X_train)

# Normalize the array by 255
X_train_normalized = X_train / 255.0

# Convert the normalized array to double type
X_train_normalized = X_train_normalized.astype(np.float64)


# Define the classical neural network
def classical_nn(x, weights, biases):
    x = tf.cast(x, tf.float64)  # Cast input to tf.float64
    
    # Reshape x to have a batch dimension if needed
    if len(x.shape) == 1:
        x = tf.expand_dims(x, axis=0)
    
    hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
    hidden_layer = tf.nn.relu(hidden_layer)
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    return output_layer


# Define the quantum circuit
def quantum_circuit(params, wires):
    qml.templates.AngleEmbedding(params, wires)
    qml.templates.StronglyEntanglingLayers(params, wires)
    return qml.expval(qml.PauliZ(0))

# Initialize the PennyLane QNode
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
@qml.qnode(dev)
def quantum_model(params, features):
    quantum_circuit(params, wires=range(num_qubits))
    return qml.expval(qml.PauliZ(0))

# Initialize the classical neural network weights and biases
num_features = X_train_normalized.shape[-1]
num_hidden = 32
num_output = 1

weights = {
    'hidden': tf.Variable(tf.random.normal([num_features, num_hidden])),
    'output': tf.Variable(tf.random.normal([num_hidden, num_output]))
}

biases = {
    'hidden': tf.Variable(tf.random.normal([num_hidden])),
    'output': tf.Variable(tf.random.normal([num_output]))
}

# Define the hybrid model
def hybrid_model(params, x):
    features = classical_nn(x, weights, biases)
    return quantum_model(params, features)

# Define the cost function
def cost(params, x, y):
    predictions = [hybrid_model(params, xi) for xi in x]
    return tf.losses.mean_squared_error(y, predictions)

# Initialize the random circuit parameters
np.random.seed(0)
num_layers = 3
params = np.random.uniform(low=0, high=2 * np.pi, size=(num_layers, num_qubits))

# Set up the optimization
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
steps = 100

# Train the hybrid model
for i in range(steps):
    with tf.GradientTape() as tape:
        loss = cost(params, X_train_normalized, y_train)
    gradients = tape.gradient(loss, list(params))
    opt.apply_gradients(zip(gradients, list(params)))

# Predict steering angles for new data
X_new = driving_data.LoadNewData()
X_new_normalized = X_new / 255.0
predictions = [hybrid_model(params, xi) for xi in X_new_normalized]
