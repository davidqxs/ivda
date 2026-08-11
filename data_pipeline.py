"""
Data pipelines for NSL-KDD and UNSW-NB15, per the paper's Section 3 and the
manuscript's Section 4.1 (Dataset and Preprocessing): one-hot encode categorical
features, min-max normalize to [0,1], PCA to 15 components (NSL-KDD's stated
optimum, Table 1 of the original manuscript).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

NSLKDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# Standard NSL-KDD attack-name -> 5-class category mapping (normal, DoS, Probe, R2L, U2R).
NSLKDD_ATTACK_TO_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
    "sendmail": "R2L", "named": "R2L", "snmpgetattack": "R2L", "snmpguess": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}


def load_nslkdd(train_path, test_path):
    train = pd.read_csv(train_path, names=NSLKDD_COLUMNS)
    test = pd.read_csv(test_path, names=NSLKDD_COLUMNS)
    for df in (train, test):
        df["category"] = df["label"].map(NSLKDD_ATTACK_TO_CATEGORY)
        unknown = df["category"].isna()
        if unknown.any():
            # attack names present in test but not in the mapping table above -
            # NSL-KDD's test set intentionally includes attack types absent
            # from train; fold any unmapped name into its likely category by
            # a conservative default of "DoS" is wrong, so instead drop them
            # and report the count (should be 0 given the mapping above is
            # the standard, complete 39-attack-type table).
            names = df.loc[unknown, "label"].unique()
            raise ValueError(f"Unmapped NSL-KDD attack labels: {names}")
    return train, test


def build_nslkdd_features(train_df, test_df, pca_dim=15, random_state=0):
    feature_cols = [c for c in NSLKDD_COLUMNS if c not in ("label", "difficulty")]
    numeric_cols = [c for c in feature_cols if c not in NSLKDD_CATEGORICAL]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), NSLKDD_CATEGORICAL),
        ("num", MinMaxScaler(), numeric_cols),
    ])
    pipe = Pipeline([("pre", pre), ("pca", PCA(n_components=pca_dim, random_state=random_state))])

    X_train_raw = train_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    X_train = pipe.fit_transform(X_train_raw)
    X_test = pipe.transform(X_test_raw)

    # re-normalize post-PCA to [0,1] per dimension (paper's Eq. 9 min-max
    # normalization is applied before PCA on raw features; PCA output is
    # not naturally bounded, and the IVDA detector generation samples
    # candidates uniformly from a bounding box, so a second min-max pass
    # on the PCA-reduced space is required for that sampling to be well
    # defined - fit bounds on train only, applied to both splits).
    post_scaler = MinMaxScaler()
    X_train = post_scaler.fit_transform(X_train)
    X_test = np.clip(post_scaler.transform(X_test), 0.0, 1.0)

    y_train = train_df["category"].values
    y_test = test_df["category"].values
    return X_train, y_train, X_test, y_test, pipe, post_scaler


UNSW_CATEGORICAL = ["proto", "service", "state"]
UNSW_DROP = ["id", "label"]  # label is redundant with attack_cat (0/1 vs name); attack_cat is the target


def load_unsw(csv_path):
    df = pd.read_csv(csv_path)
    df["attack_cat"] = df["attack_cat"].fillna("Normal").str.strip()
    # canonicalize casing/whitespace variants seen in this CSV release
    df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})
    return df


def build_unsw_features(train_df, test_df, pca_dim=15, random_state=0):
    feature_cols = [c for c in train_df.columns if c not in UNSW_DROP + ["attack_cat"]]
    numeric_cols = [c for c in feature_cols if c not in UNSW_CATEGORICAL]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), UNSW_CATEGORICAL),
        ("num", MinMaxScaler(), numeric_cols),
    ])
    pipe = Pipeline([("pre", pre), ("pca", PCA(n_components=pca_dim, random_state=random_state))])

    X_train = pipe.fit_transform(train_df[feature_cols])
    X_test = pipe.transform(test_df[feature_cols])

    post_scaler = MinMaxScaler()
    X_train = post_scaler.fit_transform(X_train)
    X_test = np.clip(post_scaler.transform(X_test), 0.0, 1.0)

    y_train = train_df["attack_cat"].values
    y_test = test_df["attack_cat"].values
    return X_train, y_train, X_test, y_test, pipe, post_scaler
