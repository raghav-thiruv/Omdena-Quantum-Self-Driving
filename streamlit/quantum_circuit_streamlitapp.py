import streamlit as st
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import circuit_drawer, plot_state_city
import matplotlib.pyplot as plt


def update_circuit_visualization(circuit, gate_operations):
    num_qubits = circuit.num_qubits
    circuit_drawer_file = "circuit_drawer.png"

    for qubit, gate_ops in enumerate(gate_operations):
        for gate in gate_ops:
            if gate == "Hadamard":
                circuit.h(qubit)
            elif gate == "Pauli-X":
                circuit.x(qubit)
            elif gate == "Pauli-Y":
                circuit.y(qubit)
            elif gate == "Pauli-Z":
                circuit.z(qubit)
            elif gate == "CNOT":
                control = (qubit + 1) % num_qubits
                circuit.cx(qubit, control)

    circuit_drawer(circuit, output='mpl', filename=circuit_drawer_file)
    image = plt.imread(circuit_drawer_file)
    plt.close()  # Close the figure to avoid overlapping images
    return image


def main():
    st.title("IBM Quantum Composer")

    # Sidebar options
    num_qubits = st.sidebar.number_input("Number of Qubits", min_value=1, max_value=5, value=2, step=1)

    # Initialize a Quantum Circuit
    circuit = QuantumCircuit(num_qubits)
    gate_operations = [[] for _ in range(num_qubits)]

    # Quantum Gates
    gate_options = ["Hadamard", "Pauli-X", "Pauli-Y", "Pauli-Z", "CNOT"]

    # Display the Quantum Circuit
    st.subheader("Quantum Circuit")
    circuit_visualization = st.empty()

    # Add Gates to the Circuit
    st.sidebar.subheader("Add Gates")

    element= True

    for qubit in range(num_qubits):
        gate_label = f"Gate - Qubit {qubit}"
        selected_gate = st.sidebar.selectbox(gate_label, gate_options, key=f"gate-{qubit}")
        selected_qubit = qubit


        if st.sidebar.button(f"Apply Gate {qubit}", key=f"apply-{qubit}"):
            
            gate_operations[selected_qubit].append(selected_gate)

            # Reset the circuit to remove the previously applied gates
            circuit = QuantumCircuit(num_qubits)

            while element:

                # Update the circuit with the selected gates
                for qubit, gate_ops in enumerate(gate_operations):
                    for gate in gate_ops:
                        if gate == "Hadamard":
                            circuit.h(qubit)
                        elif gate == "Pauli-X":
                            circuit.x(qubit)
                        elif gate == "Pauli-Y":
                            circuit.y(qubit)
                        elif gate == "Pauli-Z":
                            circuit.z(qubit)
                        elif gate == "CNOT":
                            control = (qubit + 1) % num_qubits
                            circuit.cx(qubit, control)
                    
                if st.sidebar.subheader("Finish circuit"):
                    element= False

                # Update the Circuit Visualization
                image = update_circuit_visualization(circuit, gate_operations)
                circuit_visualization.image(image, use_column_width=True)
                


    # Execute the Circuit
    st.sidebar.subheader("Execute Circuit")
    backend_options = ["qasm_simulator", "statevector_simulator", "unitary_simulator"]
    selected_backend = st.sidebar.selectbox("Select Backend", backend_options)

    if st.sidebar.button("Run Circuit"):
        # Reset the circuit to remove the previously applied gates
        circuit = QuantumCircuit(num_qubits)

        # Update the circuit with the selected gates
        for qubit, gate_ops in enumerate(gate_operations):
            for gate in gate_ops:
                if gate == "Hadamard":
                    circuit.h(qubit)
                elif gate == "Pauli-X":
                    circuit.x(qubit)
                elif gate == "Pauli-Y":
                    circuit.y(qubit)
                elif gate == "Pauli-Z":
                    circuit.z(qubit)
                elif gate == "CNOT":
                    control = (qubit + 1) % num_qubits
                    circuit.cx(qubit, control)

        backend = Aer.get_backend(selected_backend)
        job = execute(circuit, backend)
        result = job.result()

        if selected_backend == "qasm_simulator":
            counts = result.get_counts()
            st.subheader("Measurement Results")
            st.text(counts)
        elif selected_backend == "statevector_simulator":
            statevector = result.get_statevector()
            st.subheader("Final Statevector")
            st.text(statevector)
            st.subheader("Statevector Visualization")
            st.pyplot(plot_state_city(statevector))
        elif selected_backend == "unitary_simulator":
            unitary = result.get_unitary()
            st.subheader("Final Unitary")
            st.text(unitary)


if __name__ == "__main__":
    main()
