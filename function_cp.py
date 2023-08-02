# --------------------------------------------------------------------
#  Fonctions pour la partie Computer Vision
# --------------------------------------------------------------------

import os
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import time


class clr:
    start = '\033[93m' + '\033[1m'
    color = '\033[93m'
    end = '\033[0m'

# --------------------------------------------------------------------
# Fonction pour créer un dossier s'il n'existe pas déjà
# --------------------------------------------------------------------
def os_make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# --------------------------------------------------------------------
# Fonction pour rejoindre un dossier et un fichier
# --------------------------------------------------------------------
def os_path_join(folder, file):
    """Remplacement pour `os.path.join(folder, file)` sur Windows"""
    return f'{folder}/{file}'


# --------------------------------------------------------------------
# Variable pour définir l'environnement (local ou colab)
# --------------------------------------------------------------------
ENV = 'local'

# Configuration des chemins de dossiers pour les différents environnements
if ENV == 'local':
    # Développement local
    DATA_FOLDER = './Flipkart'

# Définition des chemins de dossiers pour l'environnement Colab
if ENV == 'colab':
    # Colaboratory - décommentez les 2 lignes suivantes pour connecter à votre drive
    # from google.colab import drive
    # drive.mount('/content/drive')
    DATA_FOLDER = '/content/drive/MyDrive/data/OC6'
    # OUT_FOLDER = '/content/drive/MyDrive/data/OC6'
    # IMAGE_FOLDER = '/content/drive/MyDrive/images/OC6/cv'

# Création d'une variable IMG_FOLDER pour le chemin complet du dossier d'images
IMG_FOLDER = os_path_join(DATA_FOLDER, 'Images')
OUT_FOLDER = f'{DATA_FOLDER}/output'
IMAGE_FOLDER = f'{DATA_FOLDER}/Images'
GRAPH_FOLDER = os_path_join(DATA_FOLDER, 'Graphs')  # graphique pour les diapos

# Crée les dossiers spécifiés s'ils n'existent pas déjà
os_make_dir(IMAGE_FOLDER)
os_make_dir(OUT_FOLDER)
os_make_dir(GRAPH_FOLDER)  # graphique pour les diapos
imgPath = f'{GRAPH_FOLDER}/'
print(GRAPH_FOLDER)

# --------------------------------------------------------------------
#    Police et reglages
# --------------------------------------------------------------------


font_title = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 16, }
font_title2 = {'family': 'serif', 'color': '#114b98', 'weight': 'bold', 'size': 12, }
font_title3 = {'family': 'serif', 'color': '#4F6272', 'weight': 'bold', 'size': 10, }

mycolors = ["black", "hotpink", "b", "#4CAF50"]
AllColors = ['#99ff99', '#66b3ff', '#4F6272', '#B7C3F3', '#ff9999', '#ffcc99', '#ff6666', '#DD7596', '#8EB897',
             '#c2c2f0', '#DDA0DD', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
             '#7f7f7f', '#bcbd22', '#17becf', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
             '#a65628', '#f781bf', "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
             "#000000"]

random_state = 42


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


# ---------------------------------------------------------------
#  Sauvegarde des images
# _______________________________________________________________

import matplotlib.pyplot as plt

SAVE_IMAGES = True


def fig_name_cleaning(fig_name: str) -> str:
    """Enlever les caractères interdits dans les filenames ou filepaths"""
    return (fig_name.replace(' ', '_').replace(':', '-')
            .replace('.', '-').replace('/', '_').replace('>', 'gt.')
            .replace('_\n', '').replace('\n', '').replace('<', 'lt.'))


def to_png(graph_folder, fig_name=None) -> None:
    """
    Register the current plot figure as an image in a file.
    Must call plt.show() or show image (by calling to_png() as last row in python cell)
    to apply the call 'bbox_inches=tight', to be sure to include the whole title / legend
    in the plot area.
    """

    def get_title() -> str:
        """find current plot title (or suptitle if more than one plot)"""
        if plt.gcf()._suptitle is None:  # noqa
            return plt.gca().get_title()
        else:
            return plt.gcf()._suptitle.get_text()  # noqa

    if SAVE_IMAGES:
        if fig_name is None:
            fig_name = get_title()
        elif len(fig_name) < 9:
            fig_name = f'{fig_name}_{get_title()}'

        fig_name = fig_name_cleaning(fig_name)
        print(f'"{fig_name}.png"')
        plt.gcf().savefig(
            os_path_join(f'{graph_folder}', f'{fig_name}.png'),
            bbox_inches='tight')


#  --------------------------------------------------------------------------
#     Plot histogramme
#  --------------------------------------------------------------------------

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(df, features, bins=30, figsize=(12, 7), color='grey',
                    skip_outliers=False, thresh=3, layout=(3, 3), graphName=None):
    fig = plt.figure(figsize=figsize, dpi=96)

    for i, c in enumerate(features, 1):
        ax = fig.add_subplot(*layout, i)
        if skip_outliers:
            ser = df[c][np.abs(st.zscore(df[c])) < thresh]
        else:
            ser = df[c]
        ax.hist(ser, bins=bins, color=color)
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(), color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(), color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(), color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        # ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))

    plt.tight_layout(w_pad=0.5, h_pad=0.65)
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight', dpi=96)
    # plt.show()
    return fig



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_data(X, scaler=StandardScaler(), n_comp=0.95):
    '''
    Réduction de dimension avec PCA (Analyse en Composantes Principales).

    Parameters:
    -----------
    X : array-like or dataframe, shape (n_samples, n_features)
        Les données à réduire en dimension avec PCA.

    scaler : objet standardiseur, optionnel (par défaut StandardScaler())
        L'objet standardiseur utilisé pour la standardisation des données avant PCA.

    n_comp : float ou int, optionnel (par défaut 0.95)
        Le pourcentage de la variance totale que l'on souhaite expliquer.
        Si float, il indique le pourcentage de variance à expliquer (par exemple 0.95 pour 95%).
        Si int, il indique le nombre exact de composantes à conserver.

    Returns:
    --------
    X_reduced : array-like, shape (n_samples, n_components)
        Les données réduites en dimension avec PCA.
    '''

    # Afficher la forme (shape) des données avant la réduction de dimension
    print("Shape avant réduction : ", X.shape)

    # Standardisation des données
    X_std = scaler.fit_transform(X)

    pca = PCA(n_components=X_std.shape[1], random_state=random_state)
    pca.fit(X_std)

    # Déterminer le nombre de composantes pertinentes pour expliquer 95% de la variance
    c = 0
    for i in pca.explained_variance_ratio_.cumsum():
        c += 1
        if(i >= n_comp):
            break
    X_reduced = pca.fit_transform(X_std)[:, :c]
    # c=X_reduced.shape[1]
    # Afficher la forme (shape) des données après la réduction de dimension
    print()
    print("Shape après réduction :", X_reduced.shape)
    print("Il faut {} composantes pour expliquer {:.0f}% de la variance du dataset.".format(c, n_comp*100))
    print()
    return X_reduced, c


import numpy as np
from sklearn.manifold import TSNE

def tsne_data(X, n_components=2, perplexity=30):
    """
    Applique t-SNE sur les données X.

    Paramètres :
        X (tableau ou matrice) : Données d'entrée. Il doit s'agir d'un tableau 2D ou d'une matrice où chaque ligne représente un échantillon et chaque colonne représente une caractéristique.
        n_components (entier) : Nombre de dimensions de l'espace embarqué (2 par défaut pour la visualisation).
        perplexity (nombre flottant) : Paramètre lié au nombre de voisins les plus proches utilisés dans l'algorithme (30 par défaut).

    Renvoie :
        tableau : Données intégrées dans l'espace de plus basse dimension.
    """
    # Initialise t-SNE avec les paramètres désirés
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    # Applique t-SNE et renvoie les données intégrées
    tsne_data = tsne.fit_transform(X)
    return tsne_data

# Supposons que vous ayez déjà les données transformées X_pca, les catégories réelles y_true et les clusters prédits y_pred
import pandas as pd
def visu_categories(X_pca, y_true, reduction='PCA', figsize=(15, 6)):
    time1 = time.time()
    num_labels=len(y_true.unique())
    km = KMeans(n_clusters=num_labels, init='k-means++', random_state=random_state)

    if reduction.upper() != 'PCA':
        X_trans = tsne_data(X_pca, n_components=2)
    else:
        X_trans = X_pca

    model = km.fit(X_trans)

    y_pred = km.fit_predict(X_trans)
    ARI = np.round(adjusted_rand_score(y_true, km.labels_), 4)

    X_trans = pd.DataFrame(X_trans)
    time2 = np.round(time.time() - time1, 0)
    print(clr.color + '*' * 120 + clr.end)
    print(clr.start + f'Reduction : {reduction}, ARI :  {ARI}, time : {time2}' + clr.end)
    print(clr.color + '*' * 120 + clr.end)
    print()
    # Créer une figure et spécifier les espacements entre les sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.02})

    # fig = plt.figure(figsize=figsize, gridspec_kw={'wspace': 0.3})
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    sns.scatterplot(
        ax=ax1, x=X_trans.iloc[:, 0], y=X_trans.iloc[:, 1], hue=pd.Series(y_true))
    ax1.set_title('Représentation en fonction des catégories réelles')
    ax1.legend(loc='upper left', bbox_to_anchor=(-0.5, 1),
               prop={'size': 10},)# , 'family': 'serif'
    sns.scatterplot(
        ax=ax2, x=X_trans.iloc[:, 0], y=X_trans.iloc[:, 1], hue=pd.Series(y_pred), )#, palette='Set3'
    ax2.set_title('Représentation en fonction des clusters')
    ax2.legend(loc='best', prop={'size': 10},)

    plt.show()
    return y_pred, ARI

# --------------------------------------------------------------------------------
#  Matrice de confusion
# --------------------------------------------------------------------------------
#from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

def conf_mat_transform(y_true, y_pred, corresp) :
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    corresp_cal = np.argmax(conf_mat, axis=0)
    if not corresp:
        corresp=corresp_cal
    # corresp = [3, 1, 2, 0]
    print(clr.color+'*' * 50+clr.end)
    print(clr.start+f'Correspondance calculée : {corresp_cal}'+clr.end)
    print(clr.start+f'Correspondance          : {corresp}'+clr.end)
    print()
    print(clr.color+'*' * 50+clr.end)

    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x])

    return labels['y_pred_transform']

def matrice_confusion(data, cat_var, cat_code, cv_bagofword, corresp):
    list_categories=sorted(list(set(data[cat_var])))
    num_labels=len(list_categories)
    km = KMeans(n_clusters=num_labels, init='k-means++', random_state=random_state)
    km.fit(cv_bagofword)

    y_true=data[cat_code]
    y_pred=km.labels_
    # compter les occurrences de chaque cluster
    cluster_counts: Series = pd.Series(km.labels_).value_counts()

    # print(cluster_counts)

    cls_labels_transform = conf_mat_transform(y_true, y_pred, corresp)
    conf_mat = confusion_matrix(y_true, cls_labels_transform)
    print(conf_mat)
    print(clr.color+'*' * 50+clr.end)
    print(clr.start+'Métriques de performance par classe:.'+clr.end)
    print(clr.color+'*' * 50+clr.end)
    print()
    print(classification_report(y_true, cls_labels_transform))

    # Obtenir l'accuracy
    # accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
    # Obtenir le rapport de classification (incluant l'accuracy, précision, rappel, f1-score)
    # class_report = classification_report(y_true, cls_labels_transform,
    # target_names=list_categories, output_dict=True)

    df_cm = pd.DataFrame(conf_mat, index = [label for label in list_categories],
                         columns = [i for i in range(num_labels)])

    plt.figure(figsize = (6,4))
    ax=sns.heatmap(df_cm, annot=True, fmt=".0f", cmap="Blues", annot_kws={"size": 9})
    ax.set_title('Matrice de confusion\n')
    # ax.set_xticklabels(Corresp, rotation=0)

    # display(df_cm)
    # if list_categories is not None:
    #     plt.xticks(ticks=np.arange(len(list_categories)), labels=list_categories, rotation=45)
    #     plt.yticks(ticks=np.arange(len(list_categories)), labels=list_categories, rotation=0)


