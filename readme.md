## TP Pytorch 

### Étape préliminaire
1. Télécharger ce projet :

`git clone https://gitlab.inria.fr/chxu/pytorch_exercice.git`

2. Créer un environnement pour ce projet en utilisant le fichier *pytorch.yml*
3. Execute le programme :
       `python3 main.py --experiment "faces"`

### Questions : 
1. Où se trouve les données pour le task "faces" et quels sont les données ?
2. Quel est le but du task "faces" ?
3. Quel modèle le task "faces" choisit ? 
4. Quelle fonctionne de perte que le task "faces" utilise ?
5. Quel algorithme que le task "faces" applique pour apprendre ? 
6. Où se trouve les définitions pour les arguments de ce programme ? Comprendre chaque argument avec 
leur valeur par default.

### Exercice
1. Finir la fonction *accuracy* dans le fichier *metric.py* qui retourne la précision de la prédiction
2. Évaluer la précision de modèle aussi sur le jeu de donnée de test dans le fichier *main.py*. 
   Re-Execute le programme, quelle est votre observation sur la présicion de test 
par rapport à la précision d'apprentissage ?
3. Exécute le programme en utilisant un différent batch size 
   `python3 main.py --experiment "faces" --batch_size 4`
    Quel est votre observation ?
4. Exécute le programme en utilisant un différent learning rate
    `python3 main.py --experiment "faces" --lr 1e-3`
    Quel est votre observation ? 
5. Ajouter le choix d'algorithme **Descente de gradient par mini-lot** en appelant "sgd"
6. Ajouter un jeu de donnée [Fashion-MNIST](https://pytorch.org/vision/0.8/datasets.html#fashion-mnist) dans le programme et appeler le task "fash_mnist"  
   Indice : Il faut modifier les fichiers *args.py*, *models.py*,
*loader.py* et *trainer.py*. Exécute le programme avec les arguments propres. 
7. Ajouter un jeu de donnée tabulaire [Titanic](https://www.kaggle.com/competitions/titanic/data?select=train.csv) dans le programme 
et appeler "titanic". Il faut télécharger les fichiers csv. Participer à la compétition Kaggle 