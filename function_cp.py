# --------------------------------------------------------------------
#  Fonctions pour la partie Computer Vision
# --------------------------------------------------------------------
import os
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from plot_keras_history import show_history, plot_history
# Modele
# metrics, classification_report
from sklearn.metrics import accuracy_score, adjusted_rand_score

# date
from datetime import time


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
IMG_FOLDER = os_path_join(f'{DATA_FOLDER}', 'Images')
OUT_FOLDER = f'{DATA_FOLDER}/output'
IMAGE_FOLDER = f'{DATA_FOLDER}/Images'
GRAPH_FOLDER = os_path_join(f'{DATA_FOLDER}', 'Graphs')  # graphique pour les diapos

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

SAVE_IMAGES = True


def fig_name_cleaning(fig_name: str) -> str:
    """Enlever les caractères interdits dans les filenames ou filepaths"""
    return (fig_name.replace(' ', '_').replace(':', '-')
            .replace('.', '-').replace('/', '_').replace('>', 'gt.')
            .replace('_\n', '').replace('\n', '').replace('<', 'lt.'))


def to_png(graph_folder=GRAPH_FOLDER, fig_name=None) -> None:
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

def export_base(model_name: str, df: pd.DataFrame, file_path: str = DATA_FOLDER):
    df.to_csv(f'{file_path}/{model_name}_features.csv', index=False)


import scipy.stats as st
import matplotlib.pyplot as plt


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


def copy_images(data, from_dir, dest_dir):
    """
    Copie les images d'un répertoire source vers un répertoire de destination.

    Args:
        data (pandas.DataFrame): Un DataFrame contenant les données d'images avec au moins
        les colonnes 'category1' et 'image'.
        from_dir (str): Le chemin du répertoire source où se trouvent les images.
        dest_dir (str): Le chemin du répertoire de destination où les images
        seront copiées en fonction des catégories.

    Returns:
        None
    """
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Copie des Images"):
        label = row['category1'].strip()
        chemin_source = os_path_join(from_dir, row['image'])
        dossier_label = os_path_join(dest_dir, label)
        # os.makedirs(dossier_label, exist_ok=True)
        os_make_dir(dossier_label)  # crée le repertoire si inexistant
        chemin_destination = os_path_join(dossier_label, row['image'])
        shutil.copy(chemin_source, chemin_destination)

    print(f"Copie des images vers {dest_dir} est terminée ! ")




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

    pca = PCA(n_components=min(X_std.shape[0], X_std.shape[1]), random_state=random_state)
    pca.fit(X_std)

    # Déterminer le nombre de composantes pertinentes pour expliquer 95% de la variance
    c = 0
    for i in pca.explained_variance_ratio_.cumsum():
        c += 1
        if (i >= n_comp):
            break
    X_reduced = pca.fit_transform(X_std)[:, :c]
    # c=X_reduced.shape[1]
    # Afficher la forme (shape) des données après la réduction de dimension
    print()
    print("Shape après réduction :", X_reduced.shape)
    print("Il faut {} composantes pour expliquer {:.0f}% de la variance du dataset.".format(c, n_comp * 100))
    print()
    return X_reduced, c


import numpy as np
from sklearn.manifold import TSNE


def tsne_data(X, n_components: int = 2, perplexity: int = 30) -> object:
    # Initialise t-SNE avec les paramètres désirés
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state)
    # Applique t-SNE et renvoie les données intégrées
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def get_dataframe_name(df):
    # Parcours de l'espace de noms global pour trouver le nom correspondant au DataFrame
    for name, obj in globals().items():
        if isinstance(obj, pd.DataFrame) and obj is df:
            return name
        elif isinstance(obj, np.ndarray) and obj is df:
            return name
    return None  # Si le DataFrame n'est pas trouvé


def visu_categories_kmeans(X_pca, y_true, reduction='PCA', figsize=(15, 6)):
    time1 = time.time()
    num_labels = len(y_true.unique())
    km = KMeans(n_clusters=num_labels, init='k-means++', random_state=random_state)

    if reduction.upper() != 'PCA':
        X_trans = tsne_data(X_pca, n_components=2)
    else:
        X_trans = X_pca

    # model = km.fit(X_trans)
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
    ax1.legend(loc='best', prop={'size': 11}, )  # loc='upper left', bbox_to_anchor=(-0.5, 1), 'family': 'serif'
    legend1 = ax1.legend(loc='best', prop={'size': 11})
    legend1.get_frame().set_alpha(0)  # Rendre la légende transparente
    sns.scatterplot(
        ax=ax2, x=X_trans.iloc[:, 0], y=X_trans.iloc[:, 1], hue=pd.Series(y_pred), )  # , palette='Set3'
    ax2.set_title('Représentation en fonction des clusters')
    ax2.legend(loc='best', prop={'size': 11}, )
    legend2 = ax1.legend(loc='best', prop={'size': 11})
    legend2.get_frame().set_alpha(0)  # Rendre la légende transparente
    to_png(fig_name='visu_' + reduction.lower() + '_' + get_dataframe_name(X_pca))
    # plt.show()
    return y_pred, ARI


from sklearn.mixture import GaussianMixture


def visu_categories_gmm(X_pca, y_true, reduction='PCA', figsize=(15, 6)):
    time1 = time.time()
    num_labels = len(np.unique(y_true))

    if reduction.upper() != 'PCA':
        X_trans = tsne_data(X_pca, n_components=2)  # You need to define the tsne_data function
    else:
        X_trans = X_pca

    gmm = GaussianMixture(n_components=num_labels, random_state=random_state)
    y_pred = gmm.fit_predict(X_trans)
    ARI = np.round(adjusted_rand_score(y_true, y_pred), 4)

    X_trans = pd.DataFrame(X_trans)
    time2 = np.round(time.time() - time1, 0)
    print(clr.color + '*' * 120 + clr.end)
    print(clr.start + f'Reduction : {reduction}, ARI :  {ARI}, time : {time2}' + clr.end)
    print(clr.color + '*' * 120 + clr.end)
    print()

    # Create a figure and specify spacings between subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'wspace': 0.02})

    sns.scatterplot(ax=ax1, x=X_trans.iloc[:, 0], y=X_trans.iloc[:, 1], hue=pd.Series(y_true))
    ax1.set_title('Representation based on true categories')
    ax1.legend(loc='best', prop={'size': 11})
    legend1 = ax1.legend(loc='best', prop={'size': 11})
    legend1.get_frame().set_alpha(0)

    sns.scatterplot(ax=ax2, x=X_trans.iloc[:, 0], y=X_trans.iloc[:, 1], hue=pd.Series(y_pred))
    ax2.set_title('Representation based on GMM clusters')
    ax2.legend(loc='best', prop={'size': 11})
    legend2 = ax2.legend(loc='best', prop={'size': 11})
    legend2.get_frame().set_alpha(0)

    to_png(fig_name='visu_' + reduction.lower() + '_' + get_dataframe_name(X_pca))

    return y_pred, ARI


def visu_categories(X_pca, y_true, reduction='PCA', gmm=False, figsize=(12, 6)):
    if gmm:
        visu_categories_gmm(X_pca, y_true,
                            reduction=reduction,
                            figsize=figsize)

    else:
        visu_categories_kmeans(X_pca, y_true,
                               reduction=reduction,
                               figsize=figsize)


# --------------------------------------------------------------------------------
#  Matrice de confusion
# --------------------------------------------------------------------------------

from sklearn.metrics import confusion_matrix, classification_report


def conf_mat_transform(y_true, y_pred, corresp):
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)
    corresp_cal = np.argmax(conf_mat, axis=0)
    if not corresp:
        corresp = corresp_cal
    # corresp = [3, 1, 2, 0]
    print(clr.color + '*' * 50 + clr.end)
    print(clr.start + f'Correspondance calculée : {corresp_cal}' + clr.end)
    print(clr.start + f'Correspondance          : {corresp}' + clr.end)
    print()
    print(clr.color + '*' * 50 + clr.end)

    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x: corresp[x])

    return labels['y_pred_transform']


from pandas import Series


def matrice_confusion(data, cat_var, cat_code, cv_bagofword, corresp):
    list_categories = sorted(list(set(data[cat_var])))
    num_labels = len(list_categories)
    km = KMeans(n_clusters=num_labels, init='k-means++', random_state=random_state)
    km.fit(cv_bagofword)

    y_true = data[cat_code]
    y_pred = km.labels_
    # compter les occurrences de chaque cluster
    cluster_counts: Series = pd.Series(km.labels_).value_counts()

    # print(cluster_counts)

    cls_labels_transform = conf_mat_transform(y_true, y_pred, corresp)
    conf_mat = confusion_matrix(y_true, cls_labels_transform)
    print(conf_mat)
    print(clr.color + '*' * 50 + clr.end)
    print(clr.start + 'Métriques de performance par classe:.' + clr.end)
    print(clr.color + '*' * 50 + clr.end)
    print()
    print(classification_report(y_true, cls_labels_transform))

    # Obtenir l'accuracy
    # accuracy = np.trace(conf_mat) / float(np.sum(conf_mat))
    # Obtenir le rapport de classification (incluant l'accuracy, précision, rappel, f1-score)
    # class_report = classification_report(y_true, cls_labels_transform,
    # target_names=list_categories, output_dict=True)

    df_cm = pd.DataFrame(conf_mat, index=[label for label in list_categories],
                         columns=[i for i in range(num_labels)])

    plt.figure(figsize=(6, 4))
    ax = sns.heatmap(df_cm, annot=True, fmt=".0f", cmap="Blues", annot_kws={"size": 9})
    ax.set_title('Matrice de confusion\n')
    # ax.set_xticklabels(Corresp, rotation=0)

    # display(df_cm)
    # if list_categories is not None:
    #     plt.xticks(ticks=np.arange(len(list_categories)), labels=list_categories, rotation=45)
    #     plt.yticks(ticks=np.arange(len(list_categories)), labels=list_categories, rotation=0)
    # plt.pause(0.01)
    # Calculer l'Adjusted Rand Index (ARI)
    ARI = adjusted_rand_score(y_true, cls_labels_transform)
    # Calculer l'accuracy
    accuracy = accuracy_score(y_true, cls_labels_transform)
    return ARI, accuracy


def convert_to_dataframe(data):
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, np.ndarray):
        return pd.DataFrame(data)
    else:
        raise ValueError("Entrée doit être un DataFrame ou un ndarray.")


# ---------------------------------------------------------------------------------
#   Affichage des images en projection TSNE
# --------------------------------------------------------------------------------

# Ressources utilisées pour cette partie:
# https://nextjournal.com/ml4a/image-t-sne
# Notebooks:  Mark Creasey
# https://www.kaggle.com/code/gaborvecsei/plants-t-sne/notebook
# from sklearn import manifold
import cv2

# def tsne_data(features: pd.Series) -> np.ndarray:
#     print(f'reducer t-SNE, input shape={features.shape}')
#     time1 = time.time()
#     tsne_model = TSNE(n_components=2, perplexity=30, n_iter=2000,
#                       init='random', learning_rate=200, random_state=random_state)
#     X_tsne = tsne_model.fit_transform(features)
#     time2 = np.round(time.time() - time1, 0)
#     print(f'reducer t-SNE, shape ={X_tsne.shape} time : {time2}')
#     return X_tsne


from PIL import Image


def img_resize(img, sq_size=224, interpolation=Image.ANTIALIAS):
    if type(img) == np.ndarray:
        img = Image.fromarray(img)
    w, h = img.size
    max_hw = max(h, w)
    scale = sq_size / max_hw
    new_w = sq_size if w >= h else int(w * scale)
    new_h = sq_size if h >= h else int(h * scale)
    return cv2.resize(np.asarray(img), (new_w, new_h), interpolation)


from PIL import Image
import random


def affiche_images_tsne(X_2d_data, filelist, figsize=(20, 20), sample_size=1000, filepath=IMG_FOLDER):
    """visualise images sur t-SNE
    Parameters
    ----------
    X_2d_data np.array or pd.DataFrame of shape [m, n>=2]
    filelist   list of filenames in filepath, or full path to file if filepath is None
    """
    if len(filelist) > sample_size:
        sort_order = sorted(random.sample(range(len(filelist)), sample_size))
        filelist = [filelist[i] for i in sort_order]
        X_2d_data = X_2d_data[sort_order]

    tx, ty = X_2d_data[:, 0], X_2d_data[:, 1]
    tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
    # 0 is top, so invert ty position for same form as a scatterplot
    ty = (np.max(ty) - ty) / (np.max(ty) - np.min(ty))

    width = 2000
    height = 1500
    max_dim = 100

    # Disable DecompressionBombWarning:
    # Image size (93680328 pixels) exceeds limit of 89478485 pixels
    max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 93680328 + 1
    full_image = Image.new('RGBA', (width, height))
    for filename, x, y in zip(filelist, tx, ty):
        if filepath is None:
            tile = Image.open(filename)
        else:
            tile = Image.open(f'{filepath}/{filename}')
        tile = Image.fromarray(img_resize(tile))
        # tile = Image.fromarray(img)
        rs = max(1, tile.width / max_dim, tile.height / max_dim)
        tile = tile.resize(
            (int(tile.width / rs), int(tile.height / rs)), Image.ANTIALIAS)
        full_image.paste(tile, (int((width - max_dim) * x),
                                int((height - max_dim) * y)), mask=tile.convert('RGBA'))

    Image.MAX_IMAGE_PIXELS = max_pixels
    plt.figure(figsize=figsize)
    plt.imshow(full_image)


# ---------------------------------------------------------------
#  Sauvegarde des fichiers
# --------------------------------------------------------------
import pickle


def save_pickle(data: pd.DataFrame, filename: str, dossier: str):
    if not os.path.exists(dossier):
        os.makedirs(dossier)
    chemin = os_path_join(dossier, filename)
    with open(chemin, 'wb') as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path: str):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def list_pickles_files(directory: str, prefix: str = None):
    pickle_files = [filename for filename in os.listdir(directory)
                    if filename.endswith('.pickle')]

    if prefix is not None:
        pickle_files = [filename for filename in pickle_files
                        if filename.startswith(prefix)]

    return pickle_files


def delete_folder_empty(folder_path: str):
    # Supprimer le dossier
    try:
        os.rmdir(folder_path)  # Supprime le dossier vide
        print(f"Le dossier '{folder_path}' a été supprimé avec succès.")
    except OSError as e:
        print(e)


import shutil


# Chemin du dossier à supprimer (utilisez des barres obliques inverses pour échapper les espaces)
def delete_folder(folder_path: str):
    # Supprimer le dossier et son contenu récursivement
    try:
        shutil.rmtree(folder_path)
        print(f"Le dossier '{folder_path}' et son contenu ont été supprimés avec succès.")
    except OSError as e:
        print(f"Erreur lors de la suppression du dossier '{folder_path}': {e}")


extensions_images = ['.jpg', '.png', '.jpeg']


def copy_images(data, from_dir, dest_dir):
    for index, row in tqdm(data.iterrows(), total=len(data)):
        label = row['category1'].strip()
        image_path = os_path_join(from_dir, row['image'])
        label_dir = os_path_join(dest_dir, label)
        # img = load_img(image_path, target_size=(IMGSIZE, IMGSIZE))  #
        # cv2.imread(image_path)
        os_make_dir(label_dir)
        chemin_source = os_path_join(from_dir, row['image'])  # Chemin
        chemin_destination = os_path_join(label_dir, row['image'])
        # Copie
        shutil.copy(chemin_source, chemin_destination)
    print(dest_dir)
    print("Copie des images terminée.")


def copy_images_gray(data, from_dir, dest_dir):
    for index, row in tqdm(data.iterrows(), total=len(data)):
        label = row['category1'].strip()
        # Vérification de l'extension du fichier
        if any(row['image'].lower().endswith(ext) for ext in extensions_images):
            chemin_source = os_path_join(from_dir, row['image'])  # Chemin
            image = Image.open(chemin_source)
            image_gray = image.convert("L")  # Convertir l'image en niveaux de gris
            label_dir = os_path_join(dest_dir, label)
            os_make_dir(label_dir)
            chemin_destination = os_path_join(label_dir, row['image'])
            image_gray.save(chemin_destination)  # Enregistrer l'image convertie en gris
            # shutil.copy(chemin_source, chemin_destination)# Copie
    print(dest_dir)
    print("Copie des images terminée.")


import time
from tqdm import tqdm


def features_extraction_model(model_name, model, data: pd.DataFrame, folder_image: str = IMG_FOLDER,
                              colname: str = 'image', verbose=0, use_multiprocessing=True,
                              IMGSIZE: int = 224, prefix_name: str = None):
    start_time = time.time()
    list_models = ['VGG16', 'VGG19', 'Xception', 'ResNet50']
    if model_name == 'VGG16':
        from tensorflow.keras.applications.vgg19 import preprocess_input

    elif model_name == 'VGG19':
        from tensorflow.keras.applications.vgg19 import preprocess_input

    elif model_name == 'Xception':
        from tensorflow.keras.applications.xception import preprocess_input

    elif model_name == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        raise ValueError(f"Ce modèle n'est pas pris en charge! Modèles pris en charge : {list_models}")

    pict_vectors = []
    # Boucle avec barre de progression
    for image_num in tqdm(range(data.shape[0]), desc="Processing ...", unit=colname):
        image = load_img(os_path_join(folder_image, data[colname].iloc[image_num]),
                         target_size=(IMGSIZE, IMGSIZE))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Préparer l'image pour le modèle
        image = preprocess_input(image)

        features = model.predict(image, verbose=verbose, use_multiprocessing=use_multiprocessing)
        pict_vectors.append(features.flatten())

    print('Extraction de features terminée ...!')
    end_time = time.time()
    processing_time = end_time - start_time
    print("\nTemps de traitement:", round(processing_time / 60, 2), "minutes.")
    if prefix_name:
        save_pickle(pd.DataFrame(pict_vectors), f'{model_name}_{prefix_name}_features.pickle', f'{OUT_FOLDER}')
    else:
        save_pickle(pd.DataFrame(pict_vectors), f'{model_name}_features.pickle', f'{OUT_FOLDER}')

    # Créer un DataFrame à partir des vecteurs
    return pd.DataFrame(pict_vectors), processing_time


def All_extraction_analysis(model_name, model, df_features: pd.DataFrame, data: pd.DataFrame,
                            corresp: list, colname: str = 'image', n_comp=0.99,
                            cat_code: str = 'cat_code', cat_var: str = 'category1'):
    df_features.to_csv(f'{model_name}_features.csv', index=False)
    save_pickle(df_features, f'{model_name}_features.pickle', f'{OUT_FOLDER}')

    feat_pca, c = pca_data(df_features, scaler=StandardScaler(), n_comp=n_comp)
    # y_true = data[cat_var]
    _, ARI = visu_categories(feat_pca, data[cat_var], reduction='PCA', figsize=(15, 6))
    y_pred, _ = visu_categories(feat_pca, data[cat_var], reduction='TSNE', figsize=(15, 6))

    ARI, accuracy = matrice_confusion(data, cat_var, cat_code, feat_pca, corresp)
    to_png(fig_name=f'confusion_{model_name}')

    X_tsne = tsne_data(feat_pca)
    affiche_images_tsne(X_tsne[:, 0:2], data[colname])
    plt.title(f'Visualisation TSNE des images sur la base des features {model_name}', fontsize=24)
    to_png(fig_name=f'Visualisation_TSNE_images_{model_name}')
    return ARI, accuracy


def performances_summary(model_name, ARI, accuracy, processing_time):
    # Créer un dictionnaire à partir des listes
    data = {
        'Model': [model_name],
        'ARI': [ARI],
        'Accuracy': [accuracy],
        'Time_min': [processing_time / 60]
    }
    # Créer le DataFrame
    df = pd.DataFrame(data)
    save_pickle(df, f'measures_{model_name}.pickle', f'{OUT_FOLDER}')
    return df


def learning_curve(history1, EPOCH, graphname='model1_learning_curve.png'):
    try:
        from plot_keras_history import show_history, plot_history
        import matplotlib.pyplot as plt

        show_history(history1)
        plot_history(history1, path=os_path_join(GRAPH_FOLDER, graphname))
        plt.close()
    except:
        epochs_range = range(EPOCH)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history1.history['accuracy'], label='Training Accuracy')
        plt.plot(epochs_range, history1.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history1.history['loss'], label='Training Loss')
        plt.plot(epochs_range, history1.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


# -----------------------------------------------------------------------------------------------
# Evaluation du modèle
# -----------------------------------------------------------------------------------------------


def performances_summary2(model_name, ARI, accuracy, processing_time, loss):
    # Créer un dictionnaire à partir des listes
    data = {
        'Model': [model_name],
        'ARI': [ARI],
        'Accuracy': [accuracy],
        'Loss': [loss],
        'Time_min': [processing_time / 60]
    }
    # Créer le DataFrame
    df = pd.DataFrame(data)
    save_pickle(df, f'measures_{model_name[:-3]}.pickle', f'{OUT_FOLDER}')
    return df


def evaluate(model_save_path, MODEL_NAME, history, test_generator, STEP_SIZE_TEST, processing_time):
    # start_time = time.time()

    modele = load_model(os_path_join(model_save_path, f'{MODEL_NAME}'))
    (test_loss, test_accuracy) = modele.evaluate_generator(generator=test_generator,
                                                           steps=STEP_SIZE_TEST)
    # Prédire les étiquettes sur les données de test
    y_pred_probs = modele.predict(test_generator)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test = test_generator.labels
    # Calculer l'ARI
    ari = adjusted_rand_score(y_test, y_pred_labels)

    print(clr.color + '*' * 50 + clr.end)
    print(clr.start + f'Resultats du best model :  {MODEL_NAME}' + clr.end)
    print(clr.color + '*' * 50 + clr.end)
    print("[INFO] Test Loss:      {:.4f}".format(test_loss))
    print("[INFO] Test accuracy:  {:.2f}%".format(test_accuracy * 100))
    print(clr.color + '-' * 50 + clr.end)
    show_history(history)
    plot_history(history, path=os_path_join(GRAPH_FOLDER, f'{MODEL_NAME[:-3]}_learning_curve.png'))
    plt.close()
    print('Evaluation terminée ...!')
    # end_time = time.time()
    # processing_time = end_time - start_time
    # print("\nTemps de traitement:", round(processing_time / 60, 2), "minutes.")
    performances_summary2(MODEL_NAME, ari, test_accuracy, processing_time, test_loss)
    return ari, test_accuracy, test_loss, processing_time


#  ----------------------------------------------------------------------------------
#  Afficher l'aborescence des dossiers sous python
#  __________________________________________________________________________________
#  ----------------------------------------------------------------------------------

# !apt-get install tree
# #clear_output()
# # create new folders
# # !mkdir TRAIN TEST VAL TRAIN/YES TRAIN/NO TEST/YES TEST/NO VAL/YES VAL/NO
# !tree -d

def afficher_arborescence(dossier, niveau=0):
    contenu = os.listdir(dossier)
    for element in contenu:
        chemin = os.path.join(dossier, element)
        if os.path.isdir(chemin):
            print("  " * niveau + f"📁 {element}/")
            afficher_arborescence(chemin, niveau + 1)
        else:
            print("  " * niveau + f"📄 {element}")


def afficher_schema_arborescence(dossier, prefixe=""):
    contenu = os.listdir(dossier)
    taille_contenu = len(contenu)

    for index, element in enumerate(contenu):
        est_dernier = index == taille_contenu - 1
        marqueur = "└── " if est_dernier else "├── "

        chemin = os.path.join(dossier, element)
        est_dossier = os.path.isdir(chemin)

        print(prefixe + marqueur + element)

        if est_dossier:
            nouveau_prefixe = prefixe + ("    " if est_dernier else "│   ")
            afficher_schema_arborescence(chemin, nouveau_prefixe)


