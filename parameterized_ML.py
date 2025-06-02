import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

# === Simulated Parameterized Data Generator === #
def generate_parameterized_dataset(n_samples=5000, mass_values=[500, 1000, 1500]):
    X = []
    y = []

    for mass in mass_values:
        # Signal (mass-dependent)
        lep1_pt = np.random.normal(loc=mass/10, scale=15, size=n_samples//2)
        lep2_pt = np.random.normal(loc=mass/12, scale=10, size=n_samples//2)
        lep1_eta = np.random.uniform(-2.5, 2.5, n_samples//2)
        lep1_phi = np.random.uniform(-np.pi, np.pi, n_samples//2)
        lep2_eta = np.random.uniform(-2.5, 2.5, n_samples//2)
        lep2_phi = np.random.uniform(-np.pi, np.pi, n_samples//2)
        met = np.random.normal(loc=mass/8, scale=20, size=n_samples//2)
        mass_col = np.ones(n_samples//2) * mass
        sig_data = np.vstack([lep1_pt, lep2_pt, lep1_eta, lep1_phi, lep2_eta, lep2_phi, met, mass_col]).T
        X.append(sig_data)
        y.append(np.ones(n_samples//2))

        # Background (mass-independent)
        lep1_pt = np.random.normal(loc=60, scale=20, size=n_samples//2)
        lep2_pt = np.random.normal(loc=40, scale=15, size=n_samples//2)
        lep1_eta = np.random.uniform(-2.5, 2.5, n_samples//2)
        lep1_phi = np.random.uniform(-np.pi, np.pi, n_samples//2)
        lep2_eta = np.random.uniform(-2.5, 2.5, n_samples//2)
        lep2_phi = np.random.uniform(-np.pi, np.pi, n_samples//2)
        met = np.random.normal(loc=50, scale=25, size=n_samples//2)
        mass_col = np.ones(n_samples//2) * mass  # still provide mass
        bkg_data = np.vstack([lep1_pt, lep2_pt, lep1_eta, lep1_phi, lep2_eta, lep2_phi, met, mass_col]).T
        X.append(bkg_data)
        y.append(np.zeros(n_samples//2))

    X = np.vstack(X)
    y = np.concatenate(y)
    return X, y

# === Create Dataset === #
X, y = generate_parameterized_dataset(n_samples=5000, mass_values=[500, 1000, 1500])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier === #
clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)
clf.fit(X_train, y_train)

# === Evaluation === #
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

