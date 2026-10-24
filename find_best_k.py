"""
find_best_k.py — Trouver le meilleur K pour le KNN
====================================================
Teste k=1 jusqu'à k=15 et affiche le meilleur.
"""

import csv
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Charger le dataset ────────────────────────────────────────────────────────
print("Chargement du dataset...")
with open('dataset_camera.csv', 'r') as f:
    rows = list(csv.reader(f))

X = np.array([row[:-1] for row in rows if row], dtype=float)
y = np.array([row[-1] for row in rows if row])

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset : {len(X)} exemples, {len(set(y))} classes\n")

# ── Tester k=1 jusqu'à k=15 ──────────────────────────────────────────────────
print("Test des valeurs de K :")
print("-" * 35)

best_k   = 1
best_acc = 0
results  = []

for k in range(1, 16):
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric='euclidean',
        weights='distance'  # les voisins proches comptent plus
    )
    knn.fit(X_train, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test))
    results.append((k, acc))

    marker = " ← meilleur jusqu'ici" if acc > best_acc else ""
    print(f"  k={k:2d}  →  accuracy : {acc:.1%}{marker}")

    if acc > best_acc:
        best_acc = acc
        best_k   = k

# ── Résultat final ────────────────────────────────────────────────────────────
print("-" * 35)
print(f"\n🏆 Meilleur K : {best_k}  (accuracy : {best_acc:.1%})")

# ── Réentraîner avec le meilleur K et sauvegarder ────────────────────────────
print(f"\nRéentraînement avec k={best_k} sur tout le dataset...")
best_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric='euclidean',
    weights='distance'
)
best_knn.fit(X, y)
joblib.dump(best_knn, 'model_knn_asl.pkl')
print(f"✅ Modèle sauvegardé : model_knn_asl.pkl")
print(f"   K={best_k}, weights='distance', {len(X)} exemples, {len(set(y))} classes")