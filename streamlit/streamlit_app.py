import streamlit as st
from streamlit_option_menu import option_menu

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

# Create a function for each page
def home():
    st.title("Welcome to Omdena Quantumn Self Driving Project")
    # Add content for the home page

def handle_layer_type(key, value):
    """Callback function to update layer type in the session state."""
    st.session_state[key] = value

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
    layer_types = ["CNN", "Activation", "MaxPool","Dense", "Flatten"]
    for i in range(st.session_state.num_layers):
        if f'layer_type_{i}' not in st.session_state:
            st.session_state[f'layer_type_{i}'] = "CNN"  # default layer type
        layer_type = st.selectbox(f"Layer{i+1}", layer_types,  index=layer_types.index(st.session_state[f'layer_type_{i}']))
        #st.session_state[f'layer_type_{i}'] = layer_type  # remember layer type
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


    # Add content for the build CNN page

def train_cnn():
    st.title("Train Your CNN Network")
    # Add content for the train CNN page

def view_results():
    st.title("Results")

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
st.markdown('</div>', unsafe_allow_html=True)