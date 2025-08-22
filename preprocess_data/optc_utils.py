from collections import Counter
import csv
import json
import os
import pickle as pkl
from datetime import datetime, timedelta
import re
import traceback
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import plotly.express as px
import torch.nn as nn
from scipy.stats import kendalltau
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    accuracy_score, 
)
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import silhouette_score
# import pacmap
# import hdbscan


BASE = "/srv/storage/pirat@storage3.rennes.grid5000.fr/fdijoud/"
DURATION_EVENT = 15
LEN_ENCODE_PATH = 25
WINDOW = 5

SPLIT_PATH = r"://|:/|//|/"
SPLIT_EXTENSION = re.compile(r"^(.*)\.(.*?)$")
HARD_DEVICE = r"""([/]+Device[/]+HarddiskVolume1)"""
SIMPLE_HARD_DEVICE = "C:"
SPLIT_PATH_COMMAND_LINE = r'"[^"]*"|\S+'

NODES_TYPES = ["Process"]

NODES_TYPES_ALL = [
    "Node",
    "Process",
    "File",
    "Flow",
    "Module",
    "Thread",
    "Registry",
    "Task",
    "Shell",
    "Host",
    "Service",
    "User_session",
]

LST_ACTION = [
    "CREATE",
    "DELETE",
    "MODIFY",
    "READ",
    "RENAME",
    "WRITE",
    "MESSAGE",
    "OPEN",
    "START",
    "LOAD",
    "TERMINATE",
    "ADD",
    "EDIT",
    "REMOVE",
    "COMMAND",
    "REMOTE_CREATE",
    "GRANT",
    "INTERACTIVE",
    "LOGIN",
    "LOGOUT",
    "RDP",
    "REMOTE",
    "UNLOCK",
]


LST_ACTION_END = ["DELETE", "TERMINATE", "REMOVE", "LOGOUT"]
LST_ACTION_START = ["CREATE", "START", "LOAD", "ADD", "COMMAND", "LOGIN"]


def create_folder(folder):
    """
    Create folder if not existing.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_txt(data, file):
    """
    Check if the futur saved file is not empty,
    Add extention to file name,
    Save content.
    """
    if len(data) == 0:
        print("No data")
    else:
        file_ext = file + ".txt"
        with open(file_ext, "w") as f:
            for line in data:
                f.write(line)
                f.write("\n")


def save(data, path_file):
    create_folder(os.path.dirname(path_file))
    with open(path_file, "wb") as file:
        pkl.dump(data, file)


def save_pkl(data, file):
    folder, _ = os.path.split(file)
    create_folder(folder)
    with open(file, "wb") as file:
        pkl.dump(data, file)


def load_pickle_file(file_path):
    # print(file_path)
    try:
        with open(file_path, "rb") as f:
            # print("ok")
            return pkl.load(f)
    except Exception as e:
        print(e)


def extract_csv(filename):
    pairs = []
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            pairs.append([row[0], row[1]])
        return pairs


def save_csv(filename, list1, list2):
    rows = zip(list1, list2)
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def open_json(folder, filename):
    with open(folder + filename, "r") as f:
        data = json.load(f)
    return data


def save_json(data, folder, filename):
    create_folder(folder)
    with open(folder + filename, "w") as f:
        json.dump(data, f)


def save_list_to_txt(datas, file):
    # file is the path, with the file name.
    # datas is a list to save: each element will be separated by \n in the saved file.

    # If existed file, remove it.
    if os.path.isfile(file):
        os.remove(file)

    # Create folder
    folder, filename = os.path.split(file)
    create_folder(folder)

    # Create file
    with open(file, "a") as f:
        for line in datas:
            f.write(line)
            f.write("\n")
        f.close()

    print(file)
    return


# TIME#################################
def period(start, duration):
    datetime_object_start = transform_date(start)
    datatime_object_end = datetime_object_start + timedelta(minutes=duration)
    end = datatime_object_end.isoformat(timespec="milliseconds")
    return end


def transform_date(time):
    datetimeisoformat1_re = re.compile(r"(\.\d)(-04)")
    datetimeisoformat2_re = re.compile(r"(\.\d\d)(-04)")
    if datetimeisoformat1_re.search(time):
        return datetime.fromisoformat(re.sub(r"(\.\d)(-04)", r"\g<1>00-04", time))
    elif datetimeisoformat2_re.search(time):
        return datetime.fromisoformat(re.sub(r"(\.\d\d)(-04)", r"\g<1>0-04", time))
    else:
        return datetime.fromisoformat(time)


def get_duration(start, end):
    datetime_object_start = transform_date(start)
    datetime_object_end = transform_date(end)
    duration = abs(datetime_object_end - datetime_object_start)
    total_seconds = duration.total_seconds() * 1000
    return total_seconds


def round_duration(start, duration):
    start_dt = transform_date(start)
    minutes = start_dt.minute
    rounded_minutes = str(minutes - (minutes % duration))
    if len(rounded_minutes) == 1:
        rounded_minutes = "0" + rounded_minutes
    fix = start.split(":")[0]
    rounded_start = fix + f":{rounded_minutes}:00.000-04:00"
    return rounded_start


# SETS#################################
VAL_SPLIT = "2019-09-21T00:00:00.000-04:00"


def get_train_val_files(folder):
    train_files = []
    val_files = []
    for root, dirs, files in os.walk(folder):
        if "/eval" not in root:
            train_files = [
                f"{root}/{all_files}"
                for all_files in sorted(files)
                if "2019-09-19T00:00:00.000-04:00" <= all_files < VAL_SPLIT  # graph
                # if "2019-09-20T12:00:00.000-04:00" <= all_files < VAL_SPLIT  # entities
            ]
            # train_files = train_files * 10
            val_files = [
                f"{root}/{all_files}"
                for all_files in sorted(files)
                if all_files >= VAL_SPLIT
                # if "2019-09-21T12:00:00.000-04:00" > all_files >= VAL_SPLIT  # entities
            ]
            # print(root, len(train_files))
    return train_files, val_files


def get_test_files(folder):
    test_files = []
    for root, dirs, files in os.walk(folder):
        if "/eval" in root:
            test_files = [f"{root}/{file}" for file in sorted(files)]
    return test_files

"""
# W2V#############################
def load_word2vec_model(file):
    # Return model
    try:
        model = Word2Vec.load(file)
        return model
    except Exception as e:
        print(f"Error {e}")
        exit(1)
""" 

def plot_sim(folder, x, y):

    cont = [sub[0] for sub in y]
    comb = [sub[1] for sub in y]
    geo1 = [sub[2] for sub in y]
    geo5 = [sub[3] for sub in y]
    geo25 = [sub[4] for sub in y]
    geo05 = [sub[5] for sub in y]

    plt.plot(x, geo05, label="Geo desc 0.05", alpha=0.1, color="grey")
    plt.plot(x, geo1, label="Geo desc 0.1", alpha=0.1, color="red")
    plt.plot(x, geo25, label="Geo desc 0.25", alpha=0.3, color="blue")
    plt.plot(x, geo5, label="Geo desc 0.5", alpha=0.3, color="green")
    plt.plot(x, comb, label="Combine", alpha=0.3, color="orange")
    plt.plot(x, cont, label="Constante 1", color="black")

    plt.xlabel("Pairs")
    plt.grid()
    plt.ylabel("Similarity score")
    plt.title(f"Similarity evaluation")
    plt.legend()
    plt.savefig(folder + "sim.png")
    plt.close()


def plot_sim_nberror(
    folder, simis, mistakes, places, lenght, commun, ext, q1, q2, q3, alen
):

    # Example dataset
    data = pd.DataFrame(
        {
            "Mistake": np.array(mistakes),
            "Length": np.array(lenght),
            "var-length": np.array(alen),
            "Commun": np.array(commun),
            "Q1": np.array(q1),
            "Q2-3": np.array(q2),
            "Q4": np.array(q3),
            "Ext": np.array(ext),
        }
    )

    # List1 and List2 to compare against
    cons = np.array([x[0] for x in simis])
    comb = np.array([x[1] for x in simis])
    geo1 = np.array([x[2] for x in simis])
    geo5 = np.array([x[3] for x in simis])
    geo25 = np.array([x[4] for x in simis])
    geo05 = np.array([x[5] for x in simis])

    # Calculate correlation between each variable and list1, list2
    correlations = pd.DataFrame(
        index=["Constant", "Combine", "Geo-0.05", "Geo-0.1", "Geo-0.25", "Geo-0.5"],
        columns=data.columns,
    )

    for column in data.columns:
        correlations.loc["Constant", column] = data[column].corr(
            pd.Series(cons), method="kendall"
        )
        correlations.loc["Combine", column] = data[column].corr(
            pd.Series(comb), method="kendall"
        )
        correlations.loc["Geo-0.05", column] = data[column].corr(
            pd.Series(geo05), method="kendall"
        )
        correlations.loc["Geo-0.1", column] = data[column].corr(
            pd.Series(geo1), method="kendall"
        )
        correlations.loc["Geo-0.25", column] = data[column].corr(
            pd.Series(geo25), method="kendall"
        )
        correlations.loc["Geo-0.5", column] = data[column].corr(
            pd.Series(geo5), method="kendall"
        )

    # Convert to float
    correlations = correlations.astype(float)

    # Plot the correlogram (heatmap)
    plt.figure(figsize=(10, 4))
    sns.heatmap(correlations, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("Correlation of Variables with List1 and List2")
    plt.savefig(folder + "sim-nberror.png")
    print(folder + "sim-nberror.png")
    print(pvalue_matrix_kendall(data, [cons, comb, geo1, geo05, geo25, geo5]))
    plt.close()


def pvalue_matrix_kendall(df, fonc):
    # Calculer les p-values pour chaque paire de colonnes
    for col1 in df.columns:
        for f in fonc:
            _, p_value = kendalltau(df[col1], pd.Series(f))
            print(f, col1, p_value)

    return 0

"""
def pacmac_cluser(data, sentence, labels1, labels2, folder, nb_dim):

    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    labels = clusterer.fit_predict(np.array(data))

    embedding = pacmap.PaCMAP(n_components=2)
    X_transformed = embedding.fit_transform(data, init="pca")

    # score = silhouette_score(np.array(data), labels1)
    # print(score)
    score = silhouette_score(np.array(data), labels)
    # print(score)

    # pacmac_plot(X_transformed, labels1, folder, sentence)
    # folder = folder.replace("node", "action")
    pacmac_plot(X_transformed, labels, folder, sentence)
"""

def pacmac_plot(X_embedded, labels, folder, sentence):
    da = {
        "x": X_embedded[:, 0],
        "y": X_embedded[:, 1],
        "cluster": labels,
        "name": sentence,
    }
    df = pd.DataFrame(da)

    fig = px.scatter(
        df, x="x", y="y", color="cluster", title="PACMAP", hover_name="name"
    )
    fig.write_html(folder)
    fig.show()


# ROC#############################
def roc(labelled, score, th):
    try:
        auc = roc_auc_score(labelled, score)
    except Exception as e:
        print(f"Error {e}")
        return None

    y_pred_binary = (score > th).astype(int)
    tn, fp, fn, tp = confusion_matrix(labelled, y_pred_binary).ravel()
    if tp == 0:
        preci = 0
        reca = 0
        f1 = 0
    else:
        preci = tp / (tp + fp)
        reca = tp / (tp + fn)
        f1 = 2 * preci * reca / (preci + reca)
    print(
        f"Reconstruction Results {th}.:\nTP: {tp}\nFP: {fp}\nTN: {tn}\nFN: {fn}\nPREC: {preci}\nREC: {reca}\nF1: {f1}"
    )
    return [tp, fp, tn, fn], preci, reca, f1, auc


def plot_training_curves(folder, params, number, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    create_folder(folder)

    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="x")

    plt.title(f"Training and Validation Loss.\n{params}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.savefig(folder + f"model_{number}_losse.png")
    plt.show()
    plt.close()


def plot_roc(folder, labelled, score, best_th=None, client=None):
    best_threshold = best_th
    try:
        auc = roc_auc_score(labelled, score)
    except Exception as e:
        print(f"Error {e}")
        y_pred_binary = (score > best_th).astype(int)
        fp = np.sum(y_pred_binary)
        tn = len(y_pred_binary) - fp
        print(0, fp, tn, 0)
        # print(labelled, y_pred_binary, fp, tn)
        return labelled, y_pred_binary, [0, fp, tn, 0]

    fpr, tpr, th = roc_curve(labelled, score)
    prec, rec, threshold = precision_recall_curve(labelled, score)
    print("AUC :", auc)

    youden_j = 2 * (prec * rec) / (prec + rec + 1e-10)
    best_threshold_index = np.argmax(youden_j)
    best_threshold_roc = threshold[best_threshold_index]
    if best_th is None:
        best_threshold = best_threshold_roc
    y_pred_binary = (score > best_threshold).astype(int)

    # For details
    index_des_1 = [i for i, val in enumerate(y_pred_binary) if val == 1]
    print(index_des_1, [labelled[i] for i in index_des_1])

    accuracy = accuracy_score(labelled, y_pred_binary)
    print("ACC :", accuracy)
    print("TH :", best_threshold)
    tn, fp, fn, tp = confusion_matrix(labelled, y_pred_binary).ravel()
    # print(tp, fp, tn, fn)
    if tp == 0:
        preci = 0
        reca = 0
        f1 = 0
    else:
        preci = tp / (tp + fp)
        reca = tp / (tp + fn)
        f1 = 2 * preci * reca / (preci + reca)
    if client is not None:
        print(
            f"{client} Reconstruction Results {best_threshold}.:\nTP: {tp}\nFP: {fp}\nTN: {tn}\nFN: {fn}\nPREC: {preci}\nREC: {reca}\nF1: {f1}"
        )
    else:
        print(
            f"Reconstruction Results {best_threshold}.:\nTP: {tp}\nFP: {fp}\nTN: {tn}\nFN: {fn}\nPREC: {preci}\nREC: {reca}\nF1: {f1}"
        )
        # print(
        #    f"Reconstruction Results {best_threshold}.:\ntn: {tn}\nfp: {fp}\nfn: {fn}\ntp: {tp}\nprecision: {preci}\nrecall: {reca}\nfscore: {f1}\naccuracy: {accuracy}\nauc_val: {auc}"
        # )
    if fp == 0:
        th_fpr = 0
    else:
        th_fpr = fp / (fp + tn)
    plot_auc(
        folder,
        auc,
        fpr,
        tpr,
        best_threshold_roc,
        reca,
        th_fpr,
        accuracy,
        client,
    )
    plot_prc(folder, auc, rec, prec, client)

    return labelled, y_pred_binary, [tp, fp, tn, fn]


def plot_auc(folder, auc, fpr, tpr, best_th, th_tpr, th_fpr, acc, client=None):
    plt.plot(fpr, tpr, label=f"AUC = {auc}")
    plt.scatter(
        th_fpr,
        th_tpr,
        color="red",
        marker="o",
        label=f"Best Threshold: {best_th:.4f}\nAccuracy: {acc:.2f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve result")
    plt.legend()
    if client is not None:
        create_folder(folder + "/roc")
        plt.savefig(folder + f"/roc/{client}_roc.png")
    else:
        plt.savefig(folder + f"roc.png")
    plt.close()


def plot_prc(folder, auc, rec, prec, client=None):
    plt.plot(rec, prec, label=f"AUC = {auc}")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(f"Precision Recall Curve result")
    plt.legend()
    if client is not None:
        plt.savefig(folder + f"roc/{client}_prc.png")
    else:
        plt.savefig(folder + f"prc.png")
    plt.close()


def plot_timelapse(folder, clients, f_name, loss, pred, duration):
    print(len(f_name), len(f_name[0]), len(loss), len(loss[0]), len(pred), len(pred[0]))
    # print(f_name[0], loss[0], pred[0])
    data = {}
    for i in range(len(f_name)):
        name = clients[i]
        times = []
        for j in range(len(f_name[i])):
            times.append(
                (
                    f_name[i][j],
                    period(f_name[i][j], duration),
                    loss[i][j][0],
                    loss[i][j][1],
                    pred[i][j],
                )
            )
        prediction_viz = all(t[4] == 0 for t in times)
        if prediction_viz:
            continue
        else:
            data[name] = times

    # print(data)

    # Transformer les dates en objets datetime
    for nom, periods in data.items():
        data[nom] = [
            (pd.to_datetime(start), pd.to_datetime(end), score, metric, pred)
            for start, end, score, metric, pred in periods
        ]

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(100, 35))

    # Couleur pour les boîtes
    color = ["lightcyan", "plum", "lightgreen", "lightcoral"]
    color_label = ["TN", "FN", "FP", "TP"]

    # Espacement vertical pour chaque nom
    y_start = 10  # Position de départ en ordonnées pour le premier nom
    height = 10  # Hauteur des boîtes pour chaque nom
    spacing = 5  # Espacement entre les barres (noms)

    # Définir le format de l'axe temporel
    ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=1)
    )  # Un tick toutes les heures
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))

    # Boucle à travers chaque nom et leurs périodes
    for i, nom in enumerate(tqdm(clients)):
        if nom in data:
            for start_time, end_time, score, metric, pred in data[nom]:
                # conversion datetime
                start_num = mdates.date2num(start_time)
                end_num = mdates.date2num(end_time)

                # Calcul de la durée en heures
                duration = end_num - start_num
                # Ajouter une boîte de durée 1h pour chaque période
                ax.broken_barh(
                    [(start_num, duration)],
                    (y_start + i * (height + spacing), height),
                    facecolors=color[pred],
                    edgecolors="black",
                )

                # Ajouter le score au centre de la boîte
                # ax.text(
                #    start_num + duration / 2,
                #    y_start + i * (height + spacing) + height / 2,
                #    f"{score:.3f}\n{metric[0], metric[1], metric[3]}",
                # f"{score:.3f}",
                #    ha="center",
                #    va="center",
                #    color="black",
                #    fontsize=5,
                # )
    try:
        # Définir les étiquettes de l'axe Y
        ax.set_yticks(
            [
                y_start + i * (height + spacing) + height / 2
                for i in range(len(data.keys()))
            ]
        )
        ax.set_yticklabels(data.keys())

        # Ajuster l'échelle de l'axe X en fonction des dates minimales et maximales
        start_times = [
            start for periods in data.values() for start, _, _, _, _ in periods
        ]
        end_times = [end for periods in data.values() for _, end, _, _, _ in periods]
        ax.set_xlim([min(start_times), max(end_times)])

        font_size_big = 100
        font_size_labes = 80
        # Titres et labels
        # plt.title(
        #    "Timeline graphs regarding predictions and hosts.",
        #    fontsize=font_size_big,
        # )
        plt.xlabel("Time (without time zone differences)", fontsize=font_size_big)
        plt.ylabel("Host", fontsize=font_size_big)
        plt.yticks(fontsize=font_size_labes)

        # Rotation des labels de l'axe X
        plt.xticks(rotation=45, fontsize=int(font_size_labes - font_size_labes * 0.4))

        handles = []
        for i in range(len(color)):
            handles.append(
                plt.Line2D([0], [0], color=color[i], lw=10, label=color_label[i])
            )  # Créer un rectangle vide juste pour la légende
        plt.legend(loc="upper left", handles=handles, fontsize=font_size_big)

        # Ajuster les marges pour éviter les problèmes de mise en page
        # plt.subplots_adjust(right=0.1)

        # Afficher le graphique
        plt.tight_layout()
    except Exception as e:
        print(f"Error {e}")
        traceback.print_exc()
        exit(1)

    # Afficher le graphique
    print(folder + f"timelapse_our.png")
    plt.savefig(folder + f"timelapse_our.png", dpi=300, bbox_inches="tight")
    plt.close()
    return 0

# ENCODING######################################################""
def encoding_parent_son(lst, pipw, ipw):
    if pipw == 0:
        pipw = "0"
    if ipw == 0:
        ipw = "0"
    if "\\Users\\" in pipw and "\\Users\\" in ipw:
        lst[0] += 1
    elif "\\Users\\" in pipw and "\\Users\\" not in ipw:
        lst[1] += 1
    elif "\\Users\\" not in pipw and "\\Users\\" in ipw:
        lst[2] += 1
    else:
        lst[3] += 1
    return lst


SIDS_CLASS = ["18", "19", "20", "21", "90", "96"]


def encoding_sid(lst):
    target = [0 for i in range(len(SIDS_CLASS) + 1)]
    occurrences = Counter(lst)
    result = list(occurrences.items())
    sum = 0
    for k, i in enumerate(SIDS_CLASS):
        for j in result:
            if j[0] != 0 and f"S-1-5-{i}" in j[0]:
                target[k] += j[1]
                sum += j[1]
    target[-1] = len(lst) - sum
    return target


# Model#############################################
class EarlyStopping:
    def __init__(self, patience=5, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            # self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)


class LossCustom(nn.Module):
    def __init__(self, size, node):
        super(LossCustom, self).__init__()
        self.categori_loss = nn.CrossEntropyLoss()
        self.continu_loss = nn.MSELoss()
        self.categori_loss_eval = nn.CrossEntropyLoss(reduction="none")
        self.continu_loss_eval = nn.MSELoss(reduction="none")
        self.weight_cont = 1
        self.coeff = 1

    def forward(self, outputs, inputs):
        squared_diff = (inputs - outputs) ** 2
        total_loss = self.weight_cont * squared_diff
        return self.coeff * total_loss.mean()

    def eval_(self, outputs, inputs):
        squared_diff = (inputs.detach().cpu() - outputs.detach().cpu()) ** 2
        total_loss = self.weight_cont * squared_diff
        t = np.mean(total_loss.numpy(), axis=1)
        return self.coeff * t

    def eval_device(self, outputs, inputs):
        squared_diff = (inputs - outputs) ** 2
        total_loss = self.weight_cont * squared_diff
        print(len(inputs), inputs, len(outputs), outputs, len(total_loss), total_loss)
        t = torch.mean(total_loss, dim=1)
        return self.coeff * t

    def eval(self, outputs, inputs):
        cont_loss = self.continu_loss_eval(outputs, inputs)
        total_loss = self.weight_cont * cont_loss.numpy()
        return self.coeff * np.mean(total_loss, axis=1)


class CustomDataset(Dataset):
    def __init__(self, files, node, max_=None, min_=None, labels=None, duration=15):
        self.data = []
        self.files = self.set_node_file(files, node)
        self.item_size = self.set_item_size(self.data)
        print("Item size: ", self.item_size)
        self.max_, self.min_, self.div = self.general_normalize(self.data, max_, min_)
        self.labelled = self.extract_graph_labels(labels, files, duration)

    def extract_graph_labels(self, labels, files, duration):
        labels_graphs = []
        tru = []
        if labels is None:
            return labels_graphs
        label_df = pd.read_csv(labels)
        print(files[:2])
        for file in tqdm(files):
            host = (
                "SysClient0"
                + file.split("/encoding_action_6clients/")[1].split("/old/")[0]
                + ".systemia.com"
            )
            time = os.path.basename(file).replace(".pkl", "")
            end = period(time, duration)
            times = label_df[label_df["hostname"] == host]["timestamp"]
            times = sorted(times)

            if len(times) > 0:
                label = False
                for t in times:
                    if t >= end:
                        break
                    if time <= t < end:
                        labels_graphs.append(1)
                        print(host, t)
                        tru.append(time)
                        label = True
                        break
                if not label:
                    labels_graphs.append(0)
            else:
                labels_graphs.append(0)
        print("GRAPHS: ", len(labels_graphs), len(tru), tru)
        return labels_graphs

    def set_item_size(self, files):
        return len(files[0]) - 1  # remove the label

    def set_node_file(self, files, node):
        node_files = []
        for file in tqdm(files):
            if os.path.exists(file.replace("Process", node)):
                node_files.append(file.replace("Process", node))
                with open(file.replace("Process", node), "rb") as f:
                    line = pkl.load(f)
                    # print(line)
                    # l = np.insert(line[-75:], 0, line[0])
                    # for i in range(100):
                    self.data.append(line)
        # random.shuffle(self.data)
        return node_files

    def get_files(self):
        return self.files

    def __len__(self):
        return len(self.data)

    def general_normalize(self, data, max, min):
        if max is None:
            max = np.max(np.asarray(data), axis=0)[1:]
        if min is None:
            min = np.min(np.asarray(data), axis=0)[1:]
        div = max - min
        return max, min, div

    def __getitem__(self, idx):
        line = self.data[idx]
        # self.labelled.append(line[0])
        return self.normalize(line[1:])

    def normalize(self, data):
        data = np.asarray(data)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(self.div == 0, 0, (data - self.min_) / self.div)
        return normalized


class Autoencoder(nn.Module):
    def __init__(self, size_input, dropout_rate):
        super(Autoencoder, self).__init__()
        self.encoder = self._build_encoder(size_input, dropout_rate)
        self.decoder = self._build_decoder(size_input, dropout_rate)
        self.latent_dim = size_input // 4

    def forward(self, x):
        x_ = self.encoder(x)
        y = self.decoder(x_)
        return x_, y

    def train_(
        self,
        num_epochs,
        earlystopping,
        train_loader,
        val_loader,
        initloss,
        optim,
        scheduler,
    ):
        train_losses = []
        val_losses = []
        val_loss_tot = 0.0
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.00
            for data in tqdm(train_loader):
                inputs = data
                latents, outputs = self(inputs)
                loss = initloss(outputs, inputs)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(train_loader))
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}"
            )

            self.eval()
            val_loss = 0.00
            val_loss_last = []
            with torch.no_grad():
                for data in tqdm(val_loader):
                    inputs = data
                    latents, outputs = self(inputs)
                    # self.get_output_input(inputs, outputs)
                    loss = initloss.eval_(outputs, inputs)
                    val_loss_last.extend(loss)
                    loss = np.mean(loss)
                    val_loss += loss
                val_losses.append(val_loss / len(val_loader))
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss / len(val_loader):.4f}"
                )
            val_loss_tot = val_loss

            earlystopping(train_loss / len(train_loader), self)
            scheduler.step()
            # if earlystopping.early_stop:
            #    print(f"Early Stop after epoch {epoch+1}")
            #    return train_losses, val_losses, val_loss_tot
        return train_losses, val_losses, val_loss_tot, val_loss_last

    def eval_(self, test_loader, initloss):
        self.eval()
        losses = []
        eval_loss = 0.00
        with torch.no_grad():
            for data in tqdm(test_loader):
                inputs = data
                latents, outputs = self(inputs)
                loss = initloss.eval(outputs, inputs)
                print(len(loss))
                print(np.max(loss))
                losses.extend(loss)
                eval_loss += np.mean(loss)
                # losses.append(loss.item())
            print(f"Loss: {eval_loss:.4f}")
        return eval_loss / len(test_loader), losses

    def eval_one(self, data):
        self.eval()
        with torch.no_grad():
            latents, _ = self(data)
            return latents[0]

    def _build_encoder(self, size_input, dropout_rate):
        return nn.Sequential(
            nn.Linear(size_input, size_input // 2),
            nn.ReLU(),
            nn.Linear(size_input // 2, size_input // 4),
            nn.ReLU(),
        )

    def _build_decoder(self, size_input, dropout_rate):
        return nn.Sequential(
            nn.Linear(size_input // 4, size_input // 2),
            nn.ReLU(),
            nn.Linear(size_input // 2, size_input),
        )

    def get_output_input(self, inputs, outputs):
        print(inputs)


class CustomDatasetEntitie(Dataset):
    def __init__(self, files, node, max=None, min=None, sample=1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.files = self.set_node_file(files, node)
        self.node_size_enc = self.set_item_size(node, self.files)
        self.labelled = []
        self.sample = sample
        self.data = []
        self.file_start_indices = self.compute_len_file(node, self.files, sample)
        self.total_length = self.file_start_indices[-1]
        self.max_, self.min_, self.div_ = self.general_normalize_(self.data, max, min)

    def set_item_size(self, node, files):
        len_node_encoding = 0
        with open(files[0].replace("_event", "_action"), "rb") as f:
            line = pkl.load(f)

            return len(line[0]) - 1  # remove the label

        print("Warning: not find object type length.")
        return len_node_encoding

    def set_node_file(self, files, node):
        node_files = []
        for file in files:
            if os.path.exists(
                file.replace("_event", "_entitie").replace("Process-Process", f"{node}")
            ):
                node_files.append(
                    file.replace("_event", "_entitie").replace(
                        "Process-Process", f"{node}"
                    )
                )

        return node_files
