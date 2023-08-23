# Police et reglages

from pandas import Series

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

# ---------------------------------------------------------------
#  Fonctions pour la partie NLP
# _______________________________________________________________
# ****************************************************************

import os
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from tqdm import tqdm  # Import tqdm for the progress bar
import time
import shutil


class clr:
    start = '\033[93m' + '\033[1m'
    color = '\033[93m'
    end = '\033[0m'


import sys


def is_colab_environment():
    # Vérifier si le module 'google.colab' est présent dans la liste des modules importés
    return 'google.colab' in sys.modules


# Exemple d'utilisation
if is_colab_environment():
    print("Le code s'exécute dans l'environnement Google Colab.")
else:
    print("Le code s'exécute dans un environnement local.")

if is_colab_environment():
    from google.colab import drive

    drive.mount('/content/drive')


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


# ---------------------------------------------------------------------------------
#  Repertoire de travail
#  Configuration des chemins de dossiers pour les différents environnements
# _________________________________________________________________________________

if not is_colab_environment():
    # Développement local
    DATA_FOLDER = './Flipkart'
    # Création d'une variable IMG_FOLDER pour le chemin complet du dossier d'images
    IMG_FOLDER = os_path_join(DATA_FOLDER, 'Images')
    OUT_FOLDER = f'{DATA_FOLDER}/output'
    IMAGE_FOLDER = f'{DATA_FOLDER}/Images'
    GRAPH_FOLDER = os_path_join(DATA_FOLDER, 'Graphs')  # graphique pour les diapos
    IMG_FOLDER_PROCESS = os_path_join(DATA_FOLDER, 'Images_process')  # Images traitées

# Définition des chemins de dossiers pour l'environnement Colab
if is_colab_environment():
    # Colaboratory - décommentez les 2 lignes suivantes pour connecter à votre drive
    # from google.colab import drive
    # drive.mount('/content/drive')
    DATA_FOLDER = '/content/drive/MyDrive/OC-Projet-6'

    # Création d'une variable IMG_FOLDER pour le chemin complet du dossier d'images
    IMG_FOLDER = os_path_join(DATA_FOLDER, 'Flipkart/Images')
    OUT_FOLDER = f'{DATA_FOLDER}/Flipkart/output'
    IMAGE_FOLDER = f'{DATA_FOLDER}/Flipkart/Images'
    GRAPH_FOLDER = os_path_join(DATA_FOLDER, 'Flipkart/Graphs')  # graphique pour les diapos
    IMG_FOLDER_PROCESS = os_path_join(DATA_FOLDER, 'Flipkart/Images_process')  # Images traitées

SAVE_IMAGES = True
# imgPath = f'{DATA_FOLDER}/Graphs'
# if not os.path.exists(imgPath[:-1]):
#     os.makedirs(imgPath[:-1])

# Crée les dossiers spécifiés s'ils n'existent pas déjà
os_make_dir(IMAGE_FOLDER)
os_make_dir(OUT_FOLDER)
os_make_dir(GRAPH_FOLDER)  # graphique pour les diapos
os_make_dir(IMG_FOLDER_PROCESS)
imgPath = f'{GRAPH_FOLDER}/'
print(GRAPH_FOLDER)
os.listdir(DATA_FOLDER)[:5]

# ---------------------------------------------------------------
#  Sauvegarde des images
# _______________________________________________________________
import matplotlib.pyplot as plt

SAVE_IMAGES = True


def trim_axs(axs, N):
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def fig_name_cleaning(fig_name: str) -> str:
    """Enlever les caractères interdits dans les filenames ou filepaths"""
    return (fig_name.replace(' ', '_').replace(':', '-')
            .replace('.', '-').replace('/', '_').replace('>', 'gt.')
            .replace('_\n', '').replace('\n', '').replace('<', 'lt.'))


def to_png(fig_name=None) -> None:
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
            os_path_join(f'{GRAPH_FOLDER}', f'{fig_name}.png'),
            bbox_inches='tight')


def completude(data):
    var_dict = {}

    for col in data.columns:
        var_dict[col] = []
        var_dict[col].append(round((data[col].notna().sum() / data.shape[0]) * 100, 2))
        var_dict[col].append(data[col].isna().sum())
        var_dict[col].append(round(data[col].isna().mean() * 100, 2))

    return pd.DataFrame.from_dict(data=var_dict, orient="index",
                                  columns=["Taux completion", "Nb missings", "%missings"]).sort_values(
        by="Taux completion", ascending=False)


def namestr(obj, namespace):
    ''' fonction retourne le nom en string '''
    return [name for name in namespace if namespace[name] is obj]


def namestr(obj, namespace):
    """
    fonction retourne le nom en string
    :param obj:
    :param namespace:
    :return:
    """
    return [name for name in namespace if namespace[name] is obj]


def Camembert(data, col):
    df = data[col].value_counts().reset_index()
    L = len(df[col])
    labels = list(df['index'])
    sizes = list(df[col])
    # print(labels,"\n",sizes)
    explode = Explodetuple(L)
    colors = AllColors[:L]
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    plt.tight_layout()
    plt.show()


def remplacement(data, cols, caracter1, caracter2):
    for col in cols:
        data[col] = data[col].str.lower().str.replace(caracter1, caracter2)
    return data


def remplacement2(data, cols, caracter1, caracter2):
    for col in cols:
        data[col] = data[col].str.replace(caracter1, caracter2)
    return data


def recodage(data, cols):
    for col in cols:
        data[col] = np.where((data[col].isnull() == True), "unknown", np.where(data[col] == "", "unknown", data[col]))
    return data


# Cette fonction permet de lister les modalités avec leur occurence d'unchamp qui cumule differentes modalités.
def top_words(data, cols, nb_top=100):
    count_keyword = dict()
    for index, col in data[cols].iteritems():
        if isinstance(col, float):
            continue
        for word in col.split(','):
            if word in count_keyword.keys():
                count_keyword[word] += 1
            else:
                count_keyword[word] = 1

    keyword_top = []
    for k, v in count_keyword.items():
        keyword_top.append([k, v])
    keyword_top.sort(key=lambda x: x[1], reverse=True)
    return keyword_top[:nb_top]


def Explodetuple(m):
    liste1 = []
    for t in range(m):
        if t in [0, 1]:
            liste1.append(0.1)
        else:
            liste1.append(0)
    return tuple(liste1)


def percentFreq(values):
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        # return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return '{:.1f}%({v:d})'.format(pct, v=val)

    return my_format


def repartitionTypeVar(data, figsize=(6, 3),
                       title="Repartition par types de variables"):
    df = data.dtypes.value_counts()
    L = len(df)
    labels = list(df.index)
    sizes = list(df)

    explode = Explodetuple(L)
    colors = AllColors[:L]
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct=percentFreq(df), shadow=True, startangle=0)
    ax1.axis('equal')
    plt.tight_layout()
    plt.title(label=title, fontdict=font_title2)

    # Supprimer l'appel à plt.legend()
    plt.legend()

    # plt.show()
    return fig1, ax1


def fillingRate(data, grahName=''):
    filled = data.notna().sum().sum() / (data.shape[0] * data.shape[1])
    missing = data.isna().sum().sum() / (data.shape[0] * data.shape[1])

    taux = [filled, missing]
    labels = ["%filled", "%missing"]

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.title("Taux de completion \n", fontdict=font_title)
    ax.axis("equal")
    explode = (0.1, 0)
    ax.pie(taux, explode=explode, labels=labels, autopct='%1.2f%%', shadow=True, )
    plt.legend(labels)
    if grahName != '':
        plt.savefig(imgPath + grahName, bbox_inches='tight')
        plt.show()
        # plt.close()


import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def nuageMots(data, col, figsize=(16, 12), color="white", graphName=''):
    display(Markdown('------------------------------------'))
    display(Markdown('#### Nuage de mots sur : {0}'.format(col)))
    display(Markdown('------------------------------------'))
    fig = plt.figure(1, figsize=figsize)
    ax1 = fig.add_subplot(1, 1, 1)
    # Creation de la variable text
    df = data.loc[data[col].notnull(), :]
    text = ' '.join(cat for cat in df[col])
    # Carte avec les mots: background_color="salmon"
    word_cloud = WordCloud(width=2000, height=1000, normalize_plurals=False, random_state=1,  # colormap="Pastel1",
                           collocations=False, stopwords=STOPWORDS, background_color=color, ).generate(text)
    ax1.imshow(word_cloud, interpolation="bilinear")
    # Afficher le nuage
    plt.imshow(word_cloud)
    plt.axis("off")
    if graphName != '':
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# Frequence sur des colonnes
# ----------------------------------------------------------------------------
from IPython.display import display, Markdown


def freqSimple(data, cols):
    return data[cols].unique().tolist()


def valeurUnique(data, cols):
    return data.drop_duplicates(subset=cols)[cols]


def freqSimple2(data, col_names):
    for col_name in col_names:
        effectifs = data[col_name].value_counts()
        modalites = effectifs.index  # l'index de effectifs contient les modalités
        tab = pd.DataFrame(modalites, columns=[col_name])  # création du tableau à partir des modalités
        tab["Nombre"] = effectifs.values
        tab["Frequence"] = tab["Nombre"] / len(data)  # len(data) renvoie la taille de l'échantillon
        # tab = tab.sort_values(col_name) # tri des valeurs de la variable X (croissant)
        tab["Freq. cumul"] = tab["Frequence"].cumsum()  # cumsum calcule la somme cumulée
        display(Markdown('------------------------------------'))
        display(Markdown('#### Fréquence sur la variable ***' + col_name + '***'))
        display(Markdown('------------------------------------'))
        display(tab)


#  --------------------------------------------------------------------------
#  Graphique Count (multi-graphe) sur plusieurs variables.
#  --------------------------------------------------------------------------


def numberBycategory(data, x, axis, title, labelsize, rotation, ylabsize, saturation, palette):
    g = sns.countplot(x=x, data=data, saturation=saturation, ax=axis, palette=palette)
    for i, label1 in enumerate(axis.containers):
        axis.bar_label(label1, label_type='edge', fontsize=labelsize)
    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)
    # plt.xlabel(xtitle, color='black')

    # plt.ylabel(ytitle, color='black')
    # ax.tick_params(axis='x', colors='black')
    # g.set_xticks([])
    g.set_xlabel('')
    g.set_ylabel('Nombre', size=ylabsize)
    # ax.title.set_text(title, fontdict = {'font.size':10})
    g.set_title(title, fontdict=font_title3)
    return g


def g_multi_count(data, listvar, title, ncols=2, figsize=(12, 5), labelsize=8, rotation=0,
                  ylabsize=8, saturation=1, palette='Set3', graphName=''):
    nplot = len(listvar)
    nrows = nplot // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,  # sharey=True,  sharex=True,
                            constrained_layout=True, squeeze=False, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for ax, var in zip(axs, listvar):
        numberBycategory(data, var, ax, var, labelsize, rotation, ylabsize, saturation, palette)
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title)
    if graphName != '':
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    plt.show()


#  --------------------------------------------------------------------------
#  Graphique avec une aggreg function  (multi-graphe) sur plusieurs variables.
#  --------------------------------------------------------------------------


def aggBycategory(data, x, y: float, agg_func, axis, title, labelsize, rotation,
                  xlabsize, ylabsize, palette, saturation):
    dftemp = (data.groupby(x).agg({y: agg_func}).reset_index().round(0))
    g = sns.barplot(x=x, y=y, data=dftemp, saturation=saturation, ax=axis, palette=palette)
    for label1 in axis.containers:
        axis.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in axis.get_xticklabels():
        tick.set_rotation(rotation)
    # plt.xlabel(xtitle, color='black')
    # plt.ylabel(ytitle, color='black')
    axis.tick_params(axis='x', colors='black')
    axis.tick_params(axis='y', colors='black')
    for tick in axis.xaxis.get_major_ticks():
        tick.label.set_fontsize(xlabsize)

    for tick in axis.yaxis.get_major_ticks():
        tick.label.set_fontsize(ylabsize)

    # axes.legend(prop=dict(size=10))
    #    axis.yaxis.set_tick_params(labelsize=ylabsize)
    # g.set_xticks([])
    g.set_xlabel('', size=xlabsize)
    g.set_ylabel(agg_func + ' of ' + y, size=ylabsize)
    # ax.title.set_text(title, fontdict = {'font.size':10})
    g.set_title(title + "\n", fontdict=font_title3)
    return g


#  --------------------------------------------------------------------------
# muted, pastel, coolwarm,'Accent', 'cubehelix',
# 'gist_rainbow', 'terrain', 'viridis', vlag
#  --------------------------------------------------------------------------

def g_multi_agg(data, listvar, aggvar, agg_func, title, ncols=2, figsize=(12, 5), labelsize=8,
                rotation=0, xlabsize=8, ylabsize=8, palette='coolwarm', saturation=0.85, graphName='',
                shareyy: bool = True, sharexx: bool = False):
    nplot = len(listvar)
    nrows = nplot // ncols + 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharey=shareyy, sharex=sharexx,
                            constrained_layout=True, squeeze=False, figsize=figsize)
    axs = trim_axs(axs, nplot)
    for ax, var in zip(axs, listvar):
        aggBycategory(data, var, aggvar, agg_func, ax, var, labelsize,
                      rotation, xlabsize, ylabsize, palette, saturation)
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    if graphName != '':
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    # fig.subplots_adjust(top=0.88)
    # fig.tight_layout()
    plt.show()


#  --------------------------------------------------------------------------
#     Function countplot de seaborn
#  --------------------------------------------------------------------------

def countplot2(data, x, title, xtitle, ytitle, figsize=(10, 3),
               labelsize=8, rotation=20, graphName: str = None, style='fast'):
    plt.style.use(style)  # 'fivethirtyeight', 'ggplot', 'bmh','seaborn-v0_8-whitegrid'
    fig = plt.figure(figsize=figsize)
    ax = sns.countplot(x=x, data=data, saturation=1)
    for i, label1 in enumerate(ax.containers):
        ax.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)
    plt.xlabel(xtitle, color='black')
    plt.ylabel(ytitle, color='black')
    ax.tick_params(axis='x', colors='black')
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    # plt.tight_layout()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


#  --------------------------------------------------------------------------
#     Fonction barplot de seaborn
#  --------------------------------------------------------------------------


def barplot2(data, x, y, title, xtitle, ytitle, agg_func, figsize=(10, 3),
             labelsize=8, rotation=20, graphName: str = None, style='ggplot', palette='muted'):
    dftemp = (data.groupby(x).agg({y: agg_func}).reset_index())
    dftemp = dftemp.sort_values(by=[y], ascending=False)
    plt.style.use(style)  # 'fivethirtyeight', 'ggplot', 'bmh','seaborn-v0_8-whitegrid'
    fig = plt.figure(figsize=figsize)
    ax = sns.barplot(x=x, y=y, data=dftemp, palette=palette)  # muted, pastel,coolwarm,'Accent','cubehelix',
    # 'gist_rainbow', 'terrain', 'viridis',vlag
    for i, label1 in enumerate(ax.containers):
        ax.bar_label(label1, label_type='edge', fontsize=labelsize)  # color= AllColors[i],
    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)

    plt.xlabel(xtitle, color='black')
    plt.ylabel(ytitle, color='black')
    # plt.title(title )
    ax.tick_params(axis='x', colors='black')
    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    # plt.tight_layout()
    if graphName:
        plt.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


def barplot3(pd_df, varX, varY, agg_func, title, xrotation=0, labrotation=45,
             barlabsize=8, labcolor='m', figsize=(10, 3), graphName: str = None):
    fig = plt.figure(figsize=figsize)
    pd_df = (pd_df.groupby(varX).agg({varY: agg_func}).reset_index().round(0).sort_values(by=[varX]))
    g = sns.barplot(x=varX, y=varY, data=pd_df)
    g.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    g.set(xlabel=varX, ylabel=varY)
    # x labels
    g.set_xticklabels(pd_df[varX])
    for item in g.get_xticklabels():
        item.set_rotation(xrotation)
    # bar label orientation
    for i, v in enumerate(pd_df[varY].items()):
        g.text(i, v[1], "{:}".format(v[1]), color=labcolor, va='bottom',
               rotation=labrotation, size=barlabsize)

    fig.text(0.5, 0.90, title, ha="center", fontdict=font_title2)
    plt.tight_layout()
    if graphName:
        fig.savefig(imgPath + graphName, bbox_inches='tight')
    plt.show()


# decoupage du dataset par classe
import matplotlib.colors as mcolors


def labeler(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)


def pie_chart(df, category, detache=False, figsize=(6, 6)):
    classes = list(df[category].unique())
    L = len(classes)
    if detache:
        explode = Explodetuple(L)
    else:
        explode = [0] * L

    sizes = []
    for c in classes:
        sizes.append(df.loc[df[category] == c, category].count())
    fig0, ax1 = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode,
                                       autopct=lambda pct: labeler(pct, sizes),
                                       radius=1,
                                       colors=mcolors.BASE_COLORS,
                                       # ['#0066ff','#bb66ff','#cc66ff','#dd66ff','#ee66ff','#ff66ff','#gg66ff'],
                                       startangle=90,
                                       textprops=dict(color="w"),
                                       wedgeprops=dict(width=0.7, edgecolor='w'))

    ax1.legend(wedges, classes,
               loc='center right',
               bbox_to_anchor=(1.5, 0, 0.5, 1))

    plt.text(0, 0, 'TOTAL \n{}'.format(df[category].count()),
             weight='bold', size=10, color='#52527a',
             ha='center', va='center')

    plt.setp(autotexts, size=10, weight='bold')
    ax1.axis('equal')  # Equal aspect ratio
    plt.show()


def mostFreqTags(df: pd.DataFrame, col, nb=10, others=True, normalize=False):
    """
    Compte la fréquence des n tags les plus fréquents
    return : value_counts comme un dataframe
    """
    nb = max(1, nb)
    counts_df = (df[col].value_counts(normalize=normalize)
                 .to_frame(name='freq')
                 .rename_axis(col)
                 )
    nb = min(nb, len(counts_df))
    top_n = counts_df.head(nb).copy()
    if others:
        top_n.loc['other', 'freq'] = counts_df.iloc[nb:, 0].sum()
    return top_n.reset_index()


def plot_mostFreqTags(df: pd.DataFrame, col, nb=20, others=True, normalize=False,
                      sort_values=False, palette=None,
                      ylabel=None, titre='', soustitre='', figsize=None):
    data = mostFreqTags(df, col, nb, others, normalize).copy()
    # print(data.columns.to_list())
    ax = None
    if not figsize is None:
        _, ax = plt.subplots(figsize=figsize)
    other_count = 0
    if others:
        filter_other = data[col] == 'other'
        other_count = data[filter_other]['freq'].values.sum()
        data = data[~filter_other]
    if sort_values:
        data = data.sort_values(by=col)
    if normalize:
        ax = sns.barplot(y=data[col], x=data['freq']
                                        * 100, palette=palette, ax=ax)
        ax.set_xlabel('fréquence (%)')
    else:
        ax = sns.barplot(y=data[col], x=data['freq'], palette=palette, ax=ax)
        ax.set_xlabel("nombre d'occurrences")

    autres = ''
    if others and (other_count > 0):
        if normalize:
            other_count = f'{other_count * 100:.2f} %'
        else:
            other_count = f'{int(other_count)}'
        autres = f' [Autres valeurs = {other_count}]'
    if ylabel:
        ax.set_ylabel(ylabel)
    sns.despine()
    if len(titre) > 0:
        plt.suptitle(titre, y=1.05)
    plt.title(f'{soustitre} {autres}')
    plt.tight_layout()


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
    # X_std = scaler.fit_transform(X)

    pca = PCA(n_components=X.shape[1], random_state=random_state)
    pca.fit(X)

    # Déterminer le nombre de composantes pertinentes pour expliquer 95% de la variance
    # c = 1
    # while pca.explained_variance_ratio_[0:c].cumsum()[0] < n_comp:
    #     c += 1
    c = 0
    for i in pca.explained_variance_ratio_.cumsum():
        c += 1
        if (i >= n_comp):
            break
    X_reduced = pca.fit_transform(X)[:, :c]
    # c=X_reduced.shape[1]
    # Afficher la forme (shape) des données après la réduction de dimension
    print()
    print("Shape après réduction :", X_reduced.shape)
    print("Il faut {} composantes pour expliquer {:.0f}% de la variance du dataset.".format(c, n_comp * 100))
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


def get_dataframe_name(df):
    # Parcours de l'espace de noms global pour trouver le nom correspondant au DataFrame
    for name, obj in globals().items():
        if isinstance(obj, pd.DataFrame) and obj is df:
            return name
        elif isinstance(obj, np.ndarray) and obj is df:
            return name
    return None  # Si le DataFrame n'est pas trouvé


def visu_categories(X_pca, y_true, reduction='PCA', figsize=(15, 6)):
    time1 = time.time()
    num_labels = len(y_true.unique())
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


# --------------------------------------------------------------------------------
#  Matrice de confusion
# --------------------------------------------------------------------------------
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    accuracy = accuracy_score(y_true, cls_labels_transform)
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
    return accuracy


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


import cv2


# Convertir l'image en niveaux de gris OpenCV
def img_to_gray_opencv(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
