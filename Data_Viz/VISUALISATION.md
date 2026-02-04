# MATPLOTLIB

### simple plot

- Pour tracer des courbes avec Matploltib on utilise la fonction plt.plot(), qui fonctionne aussi bien avec une liste, un numpy array ou une Pandas Series.
- Il est possible de délimiter les axes des abscisses et des ordonnées grâce à xlim et ylim, et de leur donner des titres grâce à xlabel et ylabel.
- Les Dataframes pandas permettent de tracer des courbes plus facilement et plus intuitivement, à l'aide de méthodes intégrées qui utilisent Matplotlib en background.
ex:
Par exemple : 
- trace x enfonction de y : plt.plot(x, y) 
- délimiter le graph : plt.axis([xmin, xmax, ymin, ymax])

<span style="color:girs"><em>plt.plot(t,t,':r',t,t**2,'-g',t,t**3,'--b')</em></span>

### bar plot

a fonction plt.bar permet de tracer des diagrammes en barres, verticaux avec une seule ou plusieurs séries de valeurs. Pour afficher un bar plot il suffit d'entrer dans la fonction en premier argument les positions de l'axe des abscisses sur lesquelles les barres seront centrées, et en second argument les hauteurs des barres.

plt.bar(x, y , color , width , label)
plt.xticks([1,2,3], ['un', 'deux', 'trois']) : changer abcisse 

La fonction scatter permet de tracer des nuages de points. Elle s'utilise d'une manière similaire à plot.

plt.scatter(x, y, color, marker', s=40(lataille))

### histogrammes

La fonction plt.hist permet d'afficher des histogrammes. Elle prend principalement en arguments:
- une série de valeurs (x) ,
- les bornes des valeurs à utiliser (range: par défaut (min(x), max(x)) )
- le nombre d'intervalles (bins) ou les limites explicites des intervalles.

<span style="color:girs"><em>plt.hist(x,bins)</em></span>

le paramètre rwidth permet de réduire la largeur des barres avec un espace entre elles (pourcentage entre 0 et 1), Il est possible d'afficher les fréquences probabilistes plutôt que les nombres en ordonnées en ajoutant l'argument density = True.
L'ajout de orientation = 'horizontal' permet de tracer un histogramme horizontal.PS: bins capable de gérer la taille de bar en précisant en liste par ex : [1,10,12]

<span style="color:girs"><em>plt.hist(x, density, label, orientation, histtype, rwidth, color)</em></span>

possible de faire en 2 hist en un  en mettant x = [list1,list2]

### box plot

Les boîtes à moustaches (ou boxplots) sont des graphiques très appréciés et utilisés, notamment lors d'analyses descriptives de données continues.

<span style="color:girs"><em>plt.boxplot(data) ou plt.boxplot([data1, data2])</em></span>

La méthode boxplot() des DataFrames Pandas, permet d'afficher des boîtes à moustaches pour chaque colonne indiquée dans le paramètre column, pour toutes les colonnes sinon. Le paramètre by est le plus intéressant, il permet d'afficher une boîte à moustache pour chaque modalité d'une variable qualitative.

### Camembert

Les camemberts sont des diagrammes circulaires divisés par secteurs (wedges). C'est une manière efficace de représenter l'information lorsque l'on s'intéresse principalement à la comparaison d'un secteur avec le camembert tout entier plutôt qu'entre deux secteurs.

plt.pie(x, labels) #PS:len(params) sont de même

De nombreux paramètres permettent de customiser son camembert:

- explode : liste de la même taille que la séquence de données, permet d'éloigner une ou plusieurs part du centre en indiquant de quelle fraction de rayon chaque part doit être éloignée (0 par défaut)
- colors : séquence de couleurs à utiliser pour les parts
- labeldistance : la distance des labels au centre (> 1 pour être à l'extérieur du cercle)
- autopct : une fonction (lambda) qui prend le pourcentage calculé pour une part et renvoie ce qui doit être affiché pour ce pourcentage
- pctdistance : la distance au centre à laquelle le pourcentage précédent doit être affiché (1 = sur le cercle)
- shadow = True : indique qu'il faut afficher une ombre

### Subplots et Graphiques emboîtés 

La fonction subplot prend en arguments: le nombre de lignes de la figure (numrows) , le nombre de colonnes (numcols) et le numéro du graphique sur lequel on souhaite se positionner (compris entre 1 et numrows x numcols) .

fig = plt.figure(figsize=(10,10))
plt.subplot(221) # 2ligne et 2 colonne et la le premier box
plt.subplot(222)

Cependant, il peut être utile de noter que certaines méthodes comme plot contiennent le paramètre subplots qui s'il vaut True, divise la figure en autant de graphiques que de variables présentes. Le paramètre layout permet de choisir la disposition des cellules que l'on crée.
ex : df.plot(y = ['Product1', 'Product2', 'Returns', 'Turnover'], subplots=True, layout= (2,2),
        style = ['b--', 'm:p', 'g-.', 'c-d'], figsize=(7,7));

### Ajouter du texte et des annotations

Il est possible d'ajouter du texte aux graphiques en indiquant les coordonnées où l'on veut afficher le début de celui-ci, grâce à la commande plt.text.

plt.text(x_pos,y_pos,contenu)

Pour ajouter une annotation avec une flèche descriptive pointant vers un point précis du graphique, on utilise plt.annotate.
Cette méthode prend les arguments suivants :

- le texte que l'on veut afficher.
- xy, qui indique les coordonnées où se trouve le point à annoter.
- xytext, qui indique les coordonnées du point où démarre le texte.
- arrowprops, qui sont les propriétés de la flèche d'annotation entre { } : couleur, taille de la flèche, style de flèche, etc...

Exemple : Utilisation de la commande plt.annotate
plt.annotate('Limite', xy=(1, 2), xytext=(1, 2.5), arrowprops={'facecolor':'blue'} ) affiche une flèche bleue pointant vers le point de coordonnées (1, 2) et affiche le texte 'Limite' au point (1, 2.5).

plt.plot([-3, -2, -2, -3, -3],[5, 5, 10, 10, 5],'r', alpha = 0.6)
- On crée le carré qui entoure la partie du graphique que l'on va reproduire(à partir de en bas à gauche dans sens anti horaire)
- L'argument 'alpha' donne le pourcentage d'opacité du plot (1 opaque, 0 transparent invisible)

plt.annotate('Zoom', xy=(-1.8, 7.5), xytext=(-0.5, 7.5), 
            arrowprops={'facecolor':'red'} )
- On crée la flèche rouge, avec le texte 'Zoom' dirigé vers le point (-1.8, 7.5).

plt.axes([.55, 0.4, .2, .2])
- On crée un nouveau graphique à l'intérieur du précédent,
- dont le coin en bas à gauche démarre au point (0.55, 0.4) en distance relative, 
- où 0 représente l'origine, et 1 le bout de l'axe.
- Ce graphique aura une largeur et une hauteur représentants 20% de la largeur 
- et de la hauteur du graphique d'origine.

### class and objet

To create a graph, we need a figure. The function plt.figure returns a figure, on which we can add one or more graphs(objects 'Axes'). Arguments figsize and facecolor modify the size and the background color of the figure respectively.
The function fig.add_subplot(111) returns an Axes object on which a graph can be plotted. It is the most common way to add 'Axes' to a figure. Method add_subplot adds a 'subplot' and has 3 parameters: numrows, numcols, fignum.
- numrows represents the number of lines of subplots to instantiate.
- numcols représents the number of columns of subplots to instatiate.
- fignum varies from  1to  𝑛𝑢𝑚_{𝑟𝑜𝑤𝑠}×𝑛𝑢𝑚_{𝑐𝑜𝑙𝑠} and represents the subplot number to use.

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121) # add projection=3d for 3D
ax2 = fig.add_subplot(122)
ax1.plot([0,1,2],[1,2,3],'green')
ax2.hist([1,2,2,2,3,3,4,5,5])
ax1.set_xlabel();ax1.set_ylabel()
ax2 = ax1.twinx()  # ajoute une axe x pur ax2


fig1=...
fig2=...
ax1=fig1.add_subplot..
ax2=fig2.....

ax1 = fig.add_subplot(121, sharex=True, sharey=True) # share abscisse(ordonnées) que pour même colonne(ligne)

- get_xlim : to limit the range of values on the x-axis.
- get_xticks : to modify labels on the axis.
- get_xticklabels : to give labels on the axis.

Matplotlib contains a <em>plot_date</em> function which allows to use dates as abscissa or ordinate.

# Seaborn

- Pour visualiser la distribution d'une variable quantitative, on affiche son histogramme à l'aide de la fonction <em>sns.displot()</em>.
- Pour afficher l'estimation de la densité d'une certaine variable, on peut soit utiliser la fonction <em>sns.displot()</em> avec l'argument kde=True ou bien utiliser directement la fonction sns.kdeplot().
- Pour visualiser la fonction de répartition empirique d'une variable, on utilise la fonction <em>sns.displot()</em> en égalisant l'argument kind à ecdf (kind='ecdf').
- Pour analyser la distribution d'une variable qualitative (catégorielle), la fonction <em>sns.countplot()</em> permet de générer un diagramme en barre avec le nombre d'occurences de chaque modalité.
- Pour afficher un graphique dans le notebook on utilise la fonction plt.show().

- sns.lineplot() : une fonction qui nous permet de générer un graphique en courbe.
- sns.scatterplot() : une fonction qui nous permet de générer un nuage de points.
### La fonction relplot() :¶
- La fonction sns.relplot() permet en effet de remplacer les deux fonctions ci-dessus tout en spécifiant le type de graphique souhaité à l'aide du paramètre kind.
- Les paramètres row et col de la fonction relplot() doivent prendre des variables catégorielles pour pouvoir créer des objects FacetGrid ayant les différentes modalités de ces variables présentées sur chaque ligne et/ou colonne.
- Les paramètres size et style sont utilisés pour différencier entre les différentes variables respectivement par leur taille et/ou le style avec lesquelles elles seront présentées.
- Pour afficher un graphique dans le notebook on utilise la fonction plt.show().

- sns.lmplot() permet d'afficher une courbe de régression et un nuage de points entre deux variables. On utilise cette méthode essentiellement pour vérifier une hypothèse de linéarité entre deux variables.
- sns.pairplot() permet de générer dans le même graphique, des nuages de points entre chaque paire de variables quantitatives et la distribution de chaque variable en diagonale. Cette fonction facilite l'identification des relations entre plusieurs variables quantitatives.
- sns.heatmap() permet de générer une matrice de corrélation entre des variables quantitatives. Cette méthode permet d'étudier les variables les plus corrélées entre elles.

- Pour analyser des données quantitatives en fonction de données catégorielles, on peut retrouver 3 grands types de graphiques :
        - Les nuages de points notamment avec les stripplot.
        - Les graphiques de distribution catégorielles avec les boxplot.
        - Les graphiques d'estimation catégorielles avec les countplot.

- Nous pouvons représenter ce type de graphique :
        - soit avec la fonction catplot() en spécifiant l'argument kind.
        - soit avec la fonction spécifique.

# Plotly

<em>from plotly import graph_objs as go</em>

Pour tracer des figures avec Plotly il y a une démarche à suivre :
        - Créer une grille vide avec go.Figure
        - Ajouter les courbes avec add_trace
        - Ajuster la disposition de la figure avec update_layout
Affichage de multiples graphes avec la méthode plotly.subplots.make_subplots

fig = go.Figure()
fig.add_trace(go.Scatter(x = data.world_rank,
                         y = data.teaching,
                         text=data.university_name,
                        line = dict(color='black', width=4, dash='longdashdot'))) # couleur + motif + épaisseur
fig.update_layout(title='Teaching score VS world rank of top 100 Universities',  # titre
                   xaxis_title='World Rank',   # x label
                   yaxis_title='Score')        # y label
fig.show("notebook")


from plotly.subplots import make_subplots

fig = make_subplots(rows=1, # nombres de lignes
                    cols=2, # nombres de colonnes
                     subplot_titles = ('teaching score','research score')) # titre des différents subplots
fig.add_trace(go.Scatter(x = data.world_rank,
                        y = data.teaching,
                        text=data.university_name,
                        line = dict(color='red', width=0.5, dash='dot'),
                        name = 'teaching'),
               row = 1, col =1) # Case où afficher la figure 1 
fig.add_trace(go.Scatter(x = data.world_rank,
                        y = data.research,
                        text=data.university_name,
                        line = dict(color='green', width=3, dash='longdash'),
                        name = 'research'),
               row = 1, col =2) # Case où afficher la figure 2 
fig.update_layout(title = "teaching and research score VS world ranking",
                xaxis_title='World Rank',
                yaxis_title='Score')
fig.show("notebook")

### type de graphe

go.Pie, go.Histogram, go.Scatter, px.bar, go.Box, Violin

### plotly express

import plotly.express as px
import matplotlib.pyplot as plt

plt.figure(figsize=[200,200]) # Pour déterminer la taille de la figure
fig=px.scatter(DataFrame,x='Première variable',y='Deuxième variable',animation_group='Label',..)
fig.show("notebook");