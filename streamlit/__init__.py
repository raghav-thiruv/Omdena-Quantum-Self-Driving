import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import pandas as pd
import matplotlib.pyplot as plt
import os
#from .cnn_utils import build_model, train_cnn


# Create a function for each page
def home():
    st.title("Welcome to Omdena Quantumn Self Driving Project")
    # Add content for the home page

def handle_layer_type(key, value):
    """Callback function to update layer type in the session state."""
    st.session_state[key] = value

def load_data():
    df = pd.read_csv('/Users/ansonantony/Desktop/Omdena/Quantum_Self_Driving/Omdena-Quantum-Self-Driving/Images/driving_dataset1/data.txt', names=['filename', 'steering_angle'], delimiter=' ')
    image_dir = '/Users/ansonantony/Desktop/Omdena/Quantum_Self_Driving/Omdena-Quantum-Self-Driving/Images/driving_dataset1/'
    df['filename'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

    return df

def load_image(image_path):
        try:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, (224, 224))
            image = image / 255.0  # normalize to [0,1] range
        except:
            print(f"Invalid image format, skipping: {image_path}")
            return None
        return image

def create_dataset(df):
    image_dataset = tf.data.Dataset.from_tensor_slices(df['filename'])
    angle_dataset = tf.data.Dataset.from_tensor_slices(df['steering_angle'])
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_dataset = image_dataset.apply(tf.data.experimental.ignore_errors())
    dataset = tf.data.Dataset.zip((image_dataset, angle_dataset))

    return dataset

def build_model(layers):
    model = Sequential()
    for i, layer in enumerate(layers):
        layer_type, params = layer[0], layer[1:]
        if layer_type == 'CNN':
            if i == 0:
                # specify input_shape for first layer
                model.add(Conv2D(filters=32, kernel_size=params[0], padding='same', activation='relu', input_shape=(224, 224, 3)))
            else:
                model.add(Conv2D(filters=32, kernel_size=params[0], padding='same', activation='relu'))
        # elif layer_type == 'Activation':
        #     model.add(Activation('relu'))
        elif layer_type == 'MaxPool':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        elif layer_type == 'Dense':
            model.add(Dense(units=params[0], activation='relu'))
        elif layer_type == 'Flatten':
            model.add(Flatten())

    # Add final layer with 1 output unit
    model.add(Dense(units=1))

    return model

class StreamlitCallback(Callback):
    def __init__(self, placeholder):
        super(StreamlitCallback, self).__init__()
        self.placeholder = placeholder

    def on_epoch_end(self, epoch, logs=None):
        # logs contain loss and val_loss values
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        message = f"Epoch: {epoch+1}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}"
        self.placeholder.text(message)  # Update the placeholder

def train_cnn():
    st.title("Train Your CNN Network")

    if 'model' not in st.session_state:
        st.warning("Please create a CNN first.")
        return

    model = st.session_state['model']

    # specify some training parameters
    epochs = st.number_input('Enter number of epochs', min_value=1, max_value=100, value=10)
    batch_size = st.number_input('Enter batch size', min_value=1, max_value=1000, value=32)

    model.compile(optimizer='adam', loss='mean_squared_error')  # assuming a regression problem

    # initiate early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

    # Load the data as done in the original script
    validation_split = 0.1
    df = load_data()
    df = df.sample(frac=1).reset_index(drop=True)
    val_df = df[:int(validation_split*len(df))]
    train_df = df[int(validation_split*len(df)):]

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)

    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    # Add placeholder for loss and val_loss
    loss_placeholder = st.empty()

    # Add StreamlitCallback to the list of callbacks
    callbacks = [es, mc, StreamlitCallback(loss_placeholder)]

    if st.button('Start Training'):
        # fit the model
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks, workers=4)

        # save history into session state
        st.session_state['history'] = history.history

        # Update loss placeholder
        st.experimental_rerender(loss_placeholder, max_delay=100)

        # Show success message
        st.success("Training is completed!")

    # If the user hasn't clicked the button yet, show them some instructions.
    else:
        st.info("Click the 'Start Training' button to begin training.")

    # Show success message
    st.success("Training is completed!")

def build_cnn():
    st.title("Build Your CNN Network")
    layers = []
    if 'num_layers' not in st.session_state:
        st.session_state.num_layers = 1  # default number of layers

    if st.button('Add another layer'):
        st.session_state.num_layers += 1
    if st.button('Delete last layer') and st.session_state.num_layers > 1:
        layer_to_delete = st.session_state.num_layers - 1
        if f'kernel_size_{layer_to_delete}' in st.session_state:
            del st.session_state[f'kernel_size_{layer_to_delete}']
        if f'units_{layer_to_delete}' in st.session_state:
            del st.session_state[f'units_{layer_to_delete}']
        if f'layer_type_{layer_to_delete}' in st.session_state:
            del st.session_state[f'layer_type_{layer_to_delete}']
        st.session_state.num_layers -= 1

  
    #my_form =  st.form(key="cnn")
    st.subheader("Input")
    layer_types = ["CNN", "MaxPool","Dense", "Flatten"]
    for i in range(st.session_state.num_layers):
        if f'layer_type_{i}' not in st.session_state:
            st.session_state[f'layer_type_{i}'] = "CNN"  # default layer type
        layer_type = st.selectbox(f"Layer{i+1}", layer_types,  index=layer_types.index(st.session_state[f'layer_type_{i}']))
        handle_layer_type(f'layer_type_{i}', layer_type)
        if layer_type == "CNN":
            if f'kernel_size_{i}' not in st.session_state:
                st.session_state[f'kernel_size_{i}'] = 3  # default kernel size
            kernel_size = st.number_input(f"Kernel size for CNN Layer {i+1} (enter an integer)", min_value=1, max_value=10, value=st.session_state[f'kernel_size_{i}'])
            st.session_state[f'kernel_size_{i}'] = kernel_size  # remember kernel size
            layers.append((layer_type, kernel_size))
        elif layer_type == "Dense":
            if f'units_{i}' not in st.session_state:
                st.session_state[f'units_{i}'] = 128  # default units
            units = st.number_input(f"Units for Dense Layer {i+1} (enter a positive integer)", min_value=1, max_value=1000, value=st.session_state[f'units_{i}'])
            layers.append((layer_type, units))
        else:
            if f'kernel_size_{i}' in st.session_state:
                del st.session_state[f'kernel_size_{i}']  # forget kernel size
            if f'units_{i}' in st.session_state:
                del st.session_state[f'units_{i}']  # forget units
            layers.append((layer_type,))
        #st.write(st.session_state)
    create_cnn = st.button(label = "Create CNN")
    if create_cnn:
        model = build_model(layers)
        st.session_state['model'] = model
        st.write("Model has been created. Click 'Train CNN' button to train it.")

    # Button to initiate training
    if st.button('Train CNN'):
        train_cnn()  # Assumes model has been created

    # Button to view results
    if st.button('View Results'):
        view_results(model)



def view_results():
    st.title("Results")

    # Check if history is available
    if 'history' not in st.session_state:
        st.warning("No training history found. Train a model first.")
        return

    # Retrieve history
    history = st.session_state['history']

    # Create a line plot for the loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True)

    # Display plot in Streamlit
    st.pyplot(plt)

    # Clear the figure for next render
    plt.clf()

def main():
    st.markdown(
        """
        <style>
        .navbar-container {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 20px;
        }
        .content-container {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 100vw;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    pages = {
        "Home": "home",
        "Build CNN": "build_cnn",
        "Train CNN": "train_cnn",
        "View Results": "view_results"
    }

    with st.sidebar:
        selection = option_menu("Go to", list(pages.keys()),
                            menu_icon="cast",
                            default_index=0,
                            )
    # Create the navigation menu
    # st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
    # # Create the navigation menu using option_menu
    # selection = option_menu("Go to", list(pages.keys()),
    #                         menu_icon="cast",
    #                         default_index=0,
    #                         orientation='horizontal')
    # st.markdown('</div>', unsafe_allow_html=True)

    # Center the content
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    # Map the selection to the corresponding function
    page = pages[selection]

    # Call the selected page function
    if page == "home":
        home()
    elif page == "build_cnn":
        build_cnn()
    elif page == "train_cnn":
        train_cnn()
    elif page == "view_results":
        view_results()

    # Load steering angles from text file
    df = pd.read_csv('/Users/ansonantony/Desktop/Omdena/Quantum_Self_Driving/Omdena-Quantum-Self-Driving/Images/driving_dataset1/data.txt', names=['filename', 'steering_angle'], delimiter=' ')
    image_dir = '/Users/ansonantony/Desktop/Omdena/Quantum_Self_Driving/Omdena-Quantum-Self-Driving/Images/driving_dataset1/'
    df['filename'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))

    validation_split = 0.1
    df = df.sample(frac=1).reset_index(drop=True)
    val_df = df[:int(validation_split*len(df))]
    train_df = df[int(validation_split*len(df)):]

    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)

    batch_size = 32
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()