from requirements import *

def DCCA_model ( input_shape =4096 , nb_neurons =128) :
    Xinput = Input(shape =(input_shape ,1))
    Yinput = Input(shape =(input_shape ,1))
    model = Sequential()
    model . add(Conv1D(filters =3 , kernel_size =8 , activation ="selu", padding ="same", name ="conv1"))
    model . add(Flatten())
    model . add(Dense(nb_neurons , activation ="sigmoid"))
    
    
    encoded_l = model ( Xinput )
    encoded_r = model ( Yinput )

    # Concatenate latent spaces for CCA loss
    sub_layer = Concatenate ( name ="encoded") ([ encoded_l , encoded_r ])

    # Model (Inputs, Outputs)
    dcca_model = Model ([Xinput, Yinput], sub_layer)

    return dcca_model