import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Functie pentru a incarca si redimensiona imaginile dintr-un folder
def load_and_resize_images(folder, target_shape=(255, 255)):
    # Lista pentru a stoca imaginile
    images = []
    # Lista pentru a stoca etichetele asociate imaginilor
    labels = []

    # Parcurgem fiecare subfolder din folderul dat (Audi, BMW, Mercedes)
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        # Verificam daca este un folder
        if os.path.isdir(subfolder_path):
            # Parcurgem fiecare fisier din subfolder
            for filename in os.listdir(subfolder_path):
                # Selectam doar fisierele de tip .jpg
                if filename.lower().endswith('.jpg'):
                    img_path = os.path.join(subfolder_path, filename)
                    # Citim imaginea folosind OpenCV
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Redimensionam imaginea la dimensiunea target (255x255 pixeli)
                        img = cv2.resize(img, target_shape)
                        # Adaugam imaginea in lista
                        images.append(img)
                        # Adaugam eticheta (numele subfolderului) in lista de etichete
                        labels.append(subfolder)
                    else:
                        # In cazul in care imaginea nu poate fi citita, o ignoram
                        print(f"Fisier ignorat (corupt sau imposibil de citit): {img_path}")
    return images, labels

# Functie pentru a normaliza si aplatiza imaginile (pregatire pentru modele)
def process_images(images):
    # Transformam fiecare imagine intr-un vector unidimensional (aplatizam)
    flattened_images = [img.flatten() for img in images]
    # Normalizam pixelii la valori intre 0 si 1
    normalized_images = [img / 255.0 for img in flattened_images]
    return normalized_images

# Functie pentru a obtine o imagine aleatoare si eticheta asociata dintr-un set de date
def get_random_image_from_folder(images, labels, label_encoder):
    idx = random.choice(range(len(images)))  # Selectam un index aleator
    return images[idx], label_encoder.inverse_transform([labels[idx]])[0]  # Returnam imaginea si eticheta decodata

# Functie pentru evaluarea unui model si afisarea matricii de confuzie
def evaluate_model_and_plot_cm(classifier, dataset_images, dataset_labels_encoded, dataset_name, ax, color):
    # Facem predictii cu modelul pe setul de date
    predictions = classifier.predict(dataset_images)
    print(f"\nPerformanta pe setul {dataset_name}:")
    print(classification_report(dataset_labels_encoded, predictions))  # Afisam raportul de clasificare
    print("Acuratete:", accuracy_score(dataset_labels_encoded, predictions))  # Afisam acuratetea

    # Calculam matricea de confuzie
    cm = confusion_matrix(dataset_labels_encoded, predictions)
    ax.imshow(cm, interpolation='nearest', cmap=color)  # Afisam matricea de confuzie ca imagine
    ax.set_title(f'Matrice de confuzie - {dataset_name}')
    ax.set_xticks(np.arange(len(label_encoder.classes_)))  # Etichete pentru axa X
    ax.set_yticks(np.arange(len(label_encoder.classes_)))  # Etichete pentru axa Y
    ax.set_xticklabels(label_encoder.classes_, rotation=45)  # Rotim etichetele X
    ax.set_yticklabels(label_encoder.classes_)
    ax.set_xlabel('Eticheta prezisa')
    ax.set_ylabel('Eticheta reala')

    # Adaugam text in fiecare celula a matricei
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2. else "black")

# Functie pentru a vizualiza cateva imagini din setul de testare cu etichete reale si prezise
def show_all_cases(images, true_labels_encoded, predicted_labels_encoded, label_encoder, num_images_per_case=2, num_columns=4):
    cases = {}  
    
    # Grupam imaginile dupa combinatia (eticheta reala, eticheta prezisa)
    for i in range(len(images)):
        true_label = true_labels_encoded[i]
        pred_label = predicted_labels_encoded[i]
        case_key = (true_label, pred_label)
        if case_key not in cases:
            cases[case_key] = []
        cases[case_key].append(i)
    
    total_images = sum(min(len(indices), num_images_per_case) for indices in cases.values())
    num_rows = (total_images + num_columns - 1) // num_columns  
    
    plt.figure(figsize=(15, num_rows * 3))  
    case_count = 0
    for (true_label, pred_label), indices in cases.items():
        for idx in indices[:num_images_per_case]: 
            case_count += 1
            plt.subplot(num_rows, num_columns, case_count)
            img = images[idx].reshape(255, 255, 3)  # Reconstruim imaginea
            true_label_text = label_encoder.inverse_transform([true_label])[0]
            pred_label_text = label_encoder.inverse_transform([pred_label])[0]
            plt.imshow((img * 255).astype(np.uint8))  # Convertim la valori 0-255
            plt.title(f"T: {true_label_text}\nP: {pred_label_text}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()



# Incarcam imaginile pentru antrenare, validare si testare
train_images, train_labels = load_and_resize_images('./Train')
val_images, val_labels = load_and_resize_images('./Validare')
test_images, test_labels = load_and_resize_images('./Test')

# Preprocesam imaginile (normalizare si aplatizare)
train_images = process_images(train_images)
val_images = process_images(val_images)
test_images = process_images(test_images)
print("Numarul de imagini pentru antrenare:", len(train_images))

# Codificam etichetele in format numeric
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)  # Antrenare
val_labels_encoded = label_encoder.transform(val_labels)  # Validare
test_labels_encoded = label_encoder.transform(test_labels)  # Testare

# Antrenam un model Random Forest
rf = RandomForestClassifier(n_estimators=120)
rf.fit(train_images, train_labels_encoded)

# Antrenam un model k-NN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(train_images, train_labels_encoded)

# Antrenam un model Naive Bayes
nb = GaussianNB()
nb.fit(np.concatenate((train_images, val_images)), np.concatenate((train_labels_encoded, val_labels_encoded)))

# Cream un clasificator ansamblu (Voting Classifier)
ensemble_classifier = VotingClassifier(estimators=[
    ('Random Forest', rf),
    ('k-NN', knn),
    ('Naive Bayes', nb)
], voting='hard')  # Vot majoritar

# Antrenam clasificatorul ansamblu
ensemble_classifier.fit(train_images, train_labels_encoded)

# Evaluam modelele pe setul de antrenare
fig, axs_train = plt.subplots(2, 2, figsize=(12, 12))
evaluate_model_and_plot_cm(rf, train_images, train_labels_encoded, 'Antrenare (Random Forest)', axs_train[0, 0], 'Blues')
evaluate_model_and_plot_cm(knn, train_images, train_labels_encoded, 'Antrenare (k-NN)', axs_train[0, 1], 'Greens')
evaluate_model_and_plot_cm(nb, train_images, train_labels_encoded, 'Antrenare (Naive Bayes)', axs_train[1, 0], 'Oranges')
evaluate_model_and_plot_cm(ensemble_classifier, train_images, train_labels_encoded, 'Antrenare (Ansamblu)', axs_train[1, 1], 'Purples')
plt.tight_layout()
plt.show()

# Evaluam modelele pe setul de validare
fig, axs_val = plt.subplots(2, 2, figsize=(12, 12))
evaluate_model_and_plot_cm(rf, val_images, val_labels_encoded, 'Validare (Random Forest)', axs_val[0, 0], 'Blues')
evaluate_model_and_plot_cm(knn, val_images, val_labels_encoded, 'Validare (k-NN)', axs_val[0, 1], 'Greens')
evaluate_model_and_plot_cm(nb, val_images, val_labels_encoded, 'Validare (Naive Bayes)', axs_val[1, 0], 'Oranges')
evaluate_model_and_plot_cm(ensemble_classifier, val_images, val_labels_encoded, 'Validare (Ansamblu)', axs_val[1, 1], 'Purples')
plt.tight_layout()
plt.show()

# Evaluam modelele pe setul de testare
fig, axs_test = plt.subplots(2, 2, figsize=(12, 12))
evaluate_model_and_plot_cm(rf, test_images, test_labels_encoded, 'Testare (Random Forest)', axs_test[0, 0], 'Blues')
evaluate_model_and_plot_cm(knn, test_images, test_labels_encoded, 'Testare (k-NN)', axs_test[0, 1], 'Greens')
evaluate_model_and_plot_cm(nb, test_images, test_labels_encoded, 'Testare (Naive Bayes)', axs_test[1, 0], 'Oranges')
evaluate_model_and_plot_cm(ensemble_classifier, test_images, test_labels_encoded, 'Testare (Ansamblu)', axs_test[1, 1], 'Purples')
plt.tight_layout()
plt.show()

# Comparam acuratetea modelelor
labels = ['Random Forest', 'k-NN', 'Naive Bayes', 'Ansamblu']
train_accuracies = [
    accuracy_score(train_labels_encoded, rf.predict(train_images)),
    accuracy_score(train_labels_encoded, knn.predict(train_images)),
    accuracy_score(train_labels_encoded, nb.predict(train_images)),
    accuracy_score(train_labels_encoded, ensemble_classifier.predict(train_images))
]
val_accuracies = [
    accuracy_score(val_labels_encoded, rf.predict(val_images)),
    accuracy_score(val_labels_encoded, knn.predict(val_images)),
    accuracy_score(val_labels_encoded, nb.predict(val_images)),
    accuracy_score(val_labels_encoded, ensemble_classifier.predict(val_images))
]
test_accuracies = [
    accuracy_score(test_labels_encoded, rf.predict(test_images)),
    accuracy_score(test_labels_encoded, knn.predict(test_images)),
    accuracy_score(test_labels_encoded, nb.predict(test_images)),
    accuracy_score(test_labels_encoded, ensemble_classifier.predict(test_images))
]

# Evaluam modelele pe setul de testare
test_predictions = ensemble_classifier.predict(test_images)
show_all_cases(
    images=test_images,
    true_labels_encoded=test_labels_encoded,
    predicted_labels_encoded=test_predictions,
    label_encoder=label_encoder,
    num_images_per_case=2,  # Afisam 2 imagini per caz
    num_columns=5  # Organizam în 5 coloane
)

# Grafic pentru compararea acuratetii
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(15, 6))
rects1 = ax.bar(x - width, train_accuracies, width, label='Antrenare', color='blue')
rects2 = ax.bar(x, val_accuracies, width, label='Validare', color='orange')
rects3 = ax.bar(x + width, test_accuracies, width, label='Testare', color='green')

ax.set_ylabel('Acuratete')
ax.set_title('Comparatie intre modelele antrenate')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


