from requirements import *
from cca_loss import *
from model import *
from writing_loading_traces import *
from data_generator import *
from parameters import *

############## ARGUMENTS HANDLING #############
parser = argparse.ArgumentParser()

parser.add_argument("--training", type=int, default=0, help="Activer l'entraînement (1=oui, 0=non)")
parser.add_argument("--nb_models", type=int, default=1, help="Nombre de modèles à entraîner")
parser.add_argument("--model_version", type=int, default=0, help="Which Model To Load")

args = parser.parse_args()

training_generator, validation_generator, testing_generator, m, n, d = data_generators(new_data)




# Training OR not
if args.training == 1:
    print(f"Lancement de l'entraînement de {args.nb_models} modèle(s) ...")
    all_accuracies = []  # Store accuracies for each model

    for i in range(args.nb_models):
        # Instantiation of the siamese DCCA model
        dcca_model = DCCA_model(input_shape, nb_neurons)
        optimizer = RMSprop(learning_rate=learning_rate)
        dcca_model.compile(optimizer=optimizer, loss=[cca_loss()], run_eagerly=True)
        
        # Train the model
        history = dcca_model.fit(
            x=training_generator,
            y=None,
            validation_data=validation_generator,
            epochs=epochs,
            verbose=1,
        )
        # Save the model
        dcca_model.save(f"models/dcca_model_v{i}.keras")
        
        # Prédiction sur les nouvelles traces
        new_traces = dcca_model.predict(
            testing_generator,
            verbose=1
        )
        o1 = o2 = int (new_traces.shape [1] // 2)
        # unpack ( separate ) the output of networks for view 1 and view 2
        s_hat = new_traces [: , 0: o1 ]
        m_hat = new_traces [: , o1 : o1 + o2 ]
        d_bits = bin(d[9])[2:]
        list_corr = []
        for j in range(len(s_hat)):
            list_corr.append(np.corrcoef(s_hat[j], m_hat[j])[0][1])
        med = np.median(list_corr)
        print("Median Value : ", med)
        accu = 1
        for j in range(len(list_corr)):
            if list_corr[j] > med and int(d_bits[j]) == 0:
                accu += 1
            elif list_corr[j] < med and int(d_bits[j]) == 1:
                accu += 1
        print(f"Accuracy with median method : {accu*100/len(d_bits)}%")
        all_accuracies.append(accu)
        # Separate correlation values based on the bit value
        corr_bit1 = [list_corr[j] for j in range(len(list_corr)) if d_bits[j] == '1']
        corr_bit0 = [list_corr[j] for j in range(len(list_corr)) if d_bits[j] == '0']
        indices_bit1 = [j for j in range(len(list_corr)) if d_bits[j] == '1']
        indices_bit0 = [j for j in range(len(list_corr)) if d_bits[j] == '0']
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(corr_bit1, indices_bit1, 'og', label='Bit=1', alpha = 0.7)  # Green for bits=1
        plt.plot(corr_bit0, indices_bit0, 'ob', label='Bit=0', alpha = 0.7)  # Blue for bits=0
        plt.axvline(x=med, color='r', label='Median')
        plt.legend()
        plt.xlabel("Correlation value")
        plt.ylabel("Nb pairs")
        plt.title(f"model_v{i}")
        plt.savefig(f"models/results/dissimilarity rate model_v{i}")

    # Print summary of all model accuracies
    print("\n=== Summary of Model Accuracies ===")
    for i, acc in enumerate(all_accuracies):
        print(f"Model v{i}: {acc:.2f}% with {all_accuracies[i][:10]} -> {d_bits[:10]}")
    print(f"Average accuracy: {np.mean(all_accuracies):.2f}% ± {np.std(all_accuracies):.2f}")

else:
    print("Mode test uniquement (pas d'entraînement).")
    # Loading the model with the custom Loss
    custom_objects = {"inner_cca_objective": cca_loss()}
    dcca_model = keras.models.load_model(
        f"models/dcca_model_v{args.model_version}.keras",
        custom_objects=custom_objects
    )



# Prédiction sur les nouvelles traces
new_traces = dcca_model.predict(
    testing_generator,
    verbose=1
)

o1 = o2 = int (new_traces.shape [1] // 2)

# unpack ( separate ) the output of networks for view 1 and view 2
s_hat = new_traces [: , 0: o1 ]
m_hat = new_traces [: , o1 : o1 + o2 ]

d_bits = bin(d[9])[2:]

list_corr = []
for i in range(len(s_hat)):
    list_corr.append(np.corrcoef(s_hat[i], m_hat[i])[0][1])

print(list_corr[:10])
print(d_bits[:10])

med = np.median(list_corr)
print("Median Value : ", med)
accu = 1

for i in range(len(list_corr)):
    if list_corr[i] > med and int(d_bits[i]) == 0:
        accu += 1
    elif list_corr[i] < med and int(d_bits[i]) == 1:
        accu += 1
print(f"Accuracy with median method : {accu*100/len(d_bits)}%")

# Separate correlation values based on the bit value
corr_bit1 = [list_corr[i] for i in range(len(list_corr)) if d_bits[i] == '1']
corr_bit0 = [list_corr[i] for i in range(len(list_corr)) if d_bits[i] == '0']
indices_bit1 = [i for i in range(len(list_corr)) if d_bits[i] == '1']
indices_bit0 = [i for i in range(len(list_corr)) if d_bits[i] == '0']

# Plot
plt.figure(figsize=(10, 6))
plt.plot(corr_bit1, indices_bit1, 'og', label='Bit=1', alpha = 0.7)  # Green for bits=1
plt.plot(corr_bit0, indices_bit0, 'ob', label='Bit=0', alpha = 0.7)  # Blue for bits=0
plt.axvline(x=med, color='r', label='Median')
plt.legend()
plt.xlabel("Correlation value")
plt.ylabel("Nb pairs")
plt.title(f"Collision Correlation Attack with ground-truth labels with PreProcess BigMac (Accuracy: {accu*100/len(d_bits):.2f}%)")
plt.show()

