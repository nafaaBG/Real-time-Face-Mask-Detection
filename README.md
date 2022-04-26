## Real-time Face Mask Detection with tensorflow object detection API and SSD MobileNet
![N|Solid](https://tryolabs.com/assets/blog/2020-07-09-face-mask-detection-in-street-camera-video-streams-using-ai-behind-the-curtain/2020-07-09-face-mask-detection-in-street-camera-video-streams-using-ai-behind-the-curtain.png)
## A propos de ce document
Ceci est un document markdown sur ma première tâche dans mon stage de fin d'études à Inetum Fablab. Le document résume la présentation qui a été divisée en 2 parties : présentation générale et atelier (l'atelier est l'implémentation dans la table des matières).
Auteur : 
- Nafaa BOUGRAINE [Linkedin](https://www.linkedin.com/in/nafaa-bougraine/) <nafaa.bougraine@um5r.ac.ma>  


## Table of Contents
- [Introduction](#introduction)
- [Object Detection](#object-detection)
- [SSD MobileNet Architecture](#ssd-mobilenet-architecture)
- [TensorFlow](#tensorflow)
- [Tensorflow Object Detection API](#tensorflow-object-detection-api)
- [Acquisition et preparation des donnees](#acquisition-et-preparation-des-donnees)
- [Dataset Preprocessing](#dataset-preprocessing)
- [Data balancing](#data-balancing)
- [Data Augmentation](#data-augmentation)
- [Model Selection](#model-selection)
- [Model Configuration](#model-configuration)
- [Entrainement et Evaluation](#entrainement-et-evaluation)
    + [_Training_](#Training)
    + [_Evaluation_](#evaluation)
- [Exporter le modele](#exporter-le-modele)
- [Predictions en Real Time](#predictions-en-real-time)
- 

## Introduction
Ce projet propose un moyen simple d'atteindre cet objectif en utilisant certains outils fondamentaux d'apprentissage automatique comme TensorFlow, Keras, OpenCV et Scikit-Learn. La technique proposée reconnaît avec succès le visage dans l'image ou la vidéo, puis détermine s'il porte ou non un masque. En tant qu'exécutant de tâches de surveillance, elle peut également reconnaître un visage accompagné d'un masque en mouvement ainsi que dans une vidéo. La technique atteint une excellente précision. Nous étudions les valeurs optimales des paramètres du modèle de réseau neuronal convolutif (CNN) afin d'identifier l'existence de masques avec précision sans générer de surajustement.


## Object Detection
La détection d'objets est une technologie informatique liée à la vision par ordinateur et au traitement des images, qui consiste à détecter des instances d'objets sémantiques d'une certaine classe (tels que des êtres humains, des bâtiments ou des voitures) dans des images et des vidéos numériques. Parmi les domaines bien étudiés de la détection d'objets figurent la détection des visages et la détection des piétons. La détection d'objets a des applications dans de nombreux domaines de la vision par ordinateur, notamment la recherche d'images et la vidéosurveillance.

La détection d'objets, comme le terme l'indique, est la procédure permettant de détecter les objets dans le monde réel. Par exemple, un chien, une voiture, des humains, des oiseaux, etc. Dans ce processus, nous pouvons détecter la présence de n'importe quel objet fixe avec beaucoup de facilité. Une autre grande chose qui peut être fait avec elle est que la détection de plusieurs objets dans une seule image peut être faite facilement. Par exemple, dans l'image ci-dessous, le modèle SSD a détecté un téléphone portable, un ordinateur portable, un café et des lunettes dans une seule image. Il détecte différents objets dans une seule image.
![N|Solid](https://miro.medium.com/max/1400/1*rspH1V1cBp38J-QUYtb0cg.png)

Avec l'avènement des réseaux neuronaux profonds, la détection d'objets a pris une place centrale dans le développement de la vision par ordinateur, avec de nombreux modèles développés tels que le R-CNN (Regional Convolutional Neural Network) et sa variante (Faster-RCNN), les modèles Single Shot Detectors (SSD) ainsi que le célèbre modèle You Only Look Once (YOLO) et ses nombreuses versions.
En règle générale, les modèles de détection d'objets sont classés en deux grands types d'architecture : les détecteurs d'objets à one (single) stage comme YOLO et SSD et les détecteurs d'objets à two (dual) stage comme R-CNN. 
La principale différence entre les deux est que dans les modèles de détection d'objets à deux étapes, la région d'intérêt est d'abord déterminée et la détection est ensuite effectuée uniquement sur la région d'intérêt. Cela implique que les modèles de détection d'objets en deux étapes sont généralement plus précis que les modèles en une étape, mais ils nécessitent plus de ressources informatiques et sont plus lents. La figure ci-dessous montre une représentation des deux types de modèles de détection d'objets, avec (a) une détection en two-stage et (b) un modèle de détection en two-stage.

![N|Solid](https://miro.medium.com/max/1280/0*TCuEtSXqESGWWdID.jpg)
## SSD MobileNet Architecture
L'architecture SSD est un réseau de convolution unique qui apprend à prédire les emplacements du Bounding Box et à classer ces emplacements en un seul passage. Par conséquent, SSD peut être entraîné de bout en bout. Le réseau SSD se compose d'une architecture de base (MobileNet dans ce cas) suivie de plusieurs couches de convolution :
![N|Solid](https://miro.medium.com/max/1400/1*rweWAcDJPhBfjO-H3uaBtQ.png)
En utilisant la SSD, nous n'avons besoin que d'une seule prise de vue (One single shot) pour détecter plusieurs objets dans l'image, alors que les approches basées sur le réseau de propositions régionales (RPN) telles que la série R-CNN nécessitent deux prises de vue, une pour générer des propositions de régions, une pour détecter l'objet de chaque proposition. Ainsi, le SSD est beaucoup plus rapide que les approches basées sur le RPN en deux temps (two-shot).

Cela accélère considérablement le processus. Le modèle SSD comprend quelques améliorations telles que des caractéristiques multi-scale et des boxes par défaut, ce qui lui permet d'égaler la précision du R-CNN, car il utilise des images de moindre résolution. Comme le montre l'image ci-dessous, il atteint la vitesse de traitement en temps réel avec plus de 50 FPS dans le meilleur des cas que le R-CNN et, dans certains cas, il bat même la précision du R-CNN. La précision est mesurée par le mAP, qui représente la précision des prédictions.
![N|Solid](https://miro.medium.com/max/1400/0*mdgemxIXZIx6TfKQ.png)

Une couche convolutive est une matrice appliquée aux images qui effectue une opération mathématique sur les pixels individuels pour produire de nouveaux pixels qui sont ensuite transmis comme entrée pour la couche suivante et ainsi de suite jusqu'à la fin du réseau. La dernière couche est un entier unique qui transforme la sortie de l'image en une prédiction numérique de classe qui correspond à un objet que nous essayons de prédire. Par exemple, si "1" est associé à un chat, la prédiction de la classe "1" sera un chat, tandis que "0" sera inconnu.
![N|Solid](https://miro.medium.com/max/1086/0*4FQNdzvWtzc_SR4l.gif)


Le SSD comprend 6 couches faisant 8732 prédictions et utilise la plupart de ces prédictions pour prédire ce que l'objet est finalement.
# TensorFlow
Tensorflow est une bibliothèque open-source pour le calcul numérique et l'apprentissage automatique à grande échelle qui facilite Google Brain TensorFlow, le processus d'acquisition des données, de formation des modèles, de service des prédictions et d'affinage des résultats futurs.
- Tensorflow regroupe des modèles et des algorithmes d'apprentissage automatique et d'apprentissage profond. 
- Il utilise Python comme frontal pratique et l'exécute efficacement en C++ optimisé.
- Tensorflow permet aux développeurs de créer un graphe de calculs à effectuer. 
- Chaque nœud du graphe représente une opération mathématique et chaque connexion représente des données. Ainsi, au lieu de s'occuper de détails mineurs comme la recherche de moyens appropriés pour relier la sortie d'une fonction à l'entrée d'une autre, le développeur peut se concentrer sur la logique globale de l'application.


L'équipe de recherche en intelligence artificielle d'apprentissage profond de Google, Google Brain, a développé en 2015 TensorFlow pour l'usage interne de Google. Cette bibliothèque logicielle Open-Source est utilisée par l'équipe de recherche pour effectuer plusieurs tâches importantes.

TensorFlow est actuellement la bibliothèque logicielle la plus populaire. Plusieurs applications réelles de l'apprentissage profond rendent TensorFlow populaire. En tant que bibliothèque Open-Source pour l'apprentissage profond et l'apprentissage automatique, TensorFlow trouve un rôle à jouer dans les applications basées sur le texte, la reconnaissance d'images, la recherche vocale, et bien d'autres encore. DeepFace, le système de reconnaissance d'images de Facebook, utilise TensorFlow pour la reconnaissance d'images. Il est utilisé par le Siri d'Apple pour la reconnaissance vocale. Toutes les applications Google que vous utilisez ont fait bon usage de TensorFlow pour améliorer votre expérience.
## Tensorflow Object Detection API
Avant de passer à l'API Tensorflow, comprenons ce qu'est une API.
API est l'abréviation de Application Programming Interface (interface de programmation d'applications). Une API fournit aux développeurs un ensemble d'opérations communes afin qu'ils n'aient pas à écrire du code à partir de zéro.

De même, L'API TensorFlow Object Detection est un cadre open-source construit sur TensorFlow qui facilite la construction, l'entraînement et le déploiement de modèles de détection d'objets.
Au passé, la création d'un détecteur d'objets personnalisé ressemblait à une tâche longue et difficile. Désormais, grâce à des outils tels que l'API de détection d'objets TensorFlow, nous pouvons créer des modèles fiables rapidement.
- Il existe déjà des modèles pré-entraînés dans leur cadre, appelés Model Zoo.
-  Il comprend une collection de modèles pré-entraînés sur différents ensembles de données tels que
    + COCO (Common Objects in Context) dataset, 
    + KITTI dataset
    + Open Images Dataset


#### Installation de Tensorflow Object Detection API  :

Il y a deux façons de le faire, l'une en utilisant git et l'autre en le téléchargeant manuellement :

- En utilisant Git : Ouvrir le CMD et écrire la commande :
```git clone https://github.com/tensorflow/models.git```
- Téléchargement manuelle depuis le lien suivant https://github.com/tensorflow/models

installer TensorFlow dans notre environnement en assurons que notre environnement est activé, et faites l'installation en exécutant la commande suivante :
`pip install tensorflow==2.*`
##### Télécharger et extraire TensorFlow Model Garden :
Model Garden est un répertoire officiel de TensorFlow sur github.com. Dans cette étape, nous voulons cloner ce répertoire sur notre machine locale.
Dans notre navigateur Web, on va sur [Model Garden Repo](https://github.com/tensorflow/models) et on clique sur le bouton Code afin de sélectionner la méthode de clonage qui nous convient le mieux (les options sont HTTPS, SSH ou GitHub CLI).
![N|Solid](https://i0.wp.com/neptune.ai/wp-content/uploads/cloning-method.png?resize=1024%2C645&ssl=1)
Une fois que nous avons sélectionné la méthode de clonage, nous clonons le dépôt dans notre répertoire Tensorflow local.
##### Télécharger, installer et compiler Protobuf
Par défaut, l'API de détection d'objets TensorFlow utilise Protobuf pour configurer les paramètres du modèle et de la formation, nous avons donc besoin de cette bibliothèque pour avancer.
- Depuis la page des [versions de protoc](https://github.com/protocolbuffers/protobuf/releases), on télécharge la dernière ``protoc-*-*.zip``
- Extraire le contenu du fichier téléchargé ``protoc-*-*.zip`` dans un répertoire <PATH_TO_PB> de notre choix (e.g. *C:\Program Files\Google Protobuf)*.
- Ouvrir le CMD et cd à ``tensorflow\models\research``.
- Run ``“<PATH_TO_PB>\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=.``
- Pour vérifier si cela a fonctionné correctement, allez sur object_detection/protos et assurez-vous que les fichiers .py sont présents.

#### Installation du COCO API
COCO API est une dépendance qui ne va pas directement avec l'API de détection d'objets. Nous devons l'installer séparément. L'installation manuelle de l'API COCO introduit quelques nouvelles fonctionnalités (par exemple, un ensemble de [mesures de détection et/ou de segmentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) populaires devient disponible pour l'évaluation des modèles). L'installation se déroule comme suit :
En assurant que dans notre fenêtre de Terminal, dans le répertoire de Tensorflow. Exécuter les commandes suivantes une par une :
```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git
```

#### L'installation de Object Detection API 
C'est la dernière étape de notre bloc d'installation et de configuration ! Nous allons installer l'API de détection d'objets elle-même. Pour ce faire, nous devons installer le package object_detection. Voici comment procéder :
- Copie le fichier setup.py depuis tensorflow\research\object_detection\packages\tf2 à tensorflow\models\research\ en exécutant la commande suivante : ``cp object_detection/packages/tf2/setup.py``.
- Ouvrir le cmd et cd à ``tensorflow\models\research``.
- Run ``python -m pip install .``
- Testons si notre installation est réussie en exécutant la commande suivante à partir du répertoire Tensorflow/models/research dans notre fenêtre Terminal : python ``object_detection/builders/model_builder_tf2_test.py``

## Acquisition et preparation des donnees
Tout d'abord, pour construire notre Face mask detector, nous avons besoin de données pertinentes. De plus, en raison de la nature de notre modèle, nous avons besoin de données annotées avec des bounding box.
J'ai choisi le jeu de données ["Face Mask Detection"](https://www.kaggle.com/andrewmvd/face-mask-detection) de Kaggle et je l'ai téléchargé directement sur mon Google Drive. Le dataset téléchargé se compose de deux dossiers :
- **images**, qui comprend 853 fichiers .png
- **annotations**, qui comprend 853 fichiers .xml

[![image](https://www.linkpicture.com/q/map_11.jpg)](https://www.linkpicture.com/view.php?img=LPic6262730ba6459215022302)

#### Data balancing
L'ensemble de données est légèrement déséquilibré, ayant plus d'étiquettes with_mask, donc quelque chose que nous pouvons faire est d'augmenter avec des images d'autres classes mask_weared_incorrect et without_mask.

Donc ma solution c'était de collecter des images pour les deux classes et les annoter manuellement en utilisant LabelImg tool pour faire équilibrer un peu entre les 3 classes avant d'appliquer la technique d'augmentation pour augmenter le nombre des images.
[![image](https://www.linkpicture.com/q/labelimg.jpg)](https://www.linkpicture.com/view.php?img=LPic62614a4d0493a793309301)
## Data Augmentation
La performance de la plupart des modèles ML, et des modèles d'apprentissage profond en particulier, dépend de la qualité, de la quantité et de la pertinence des données d'entraînement. Cependant, l'insuffisance des données est l'un des défis les plus courants dans la mise en œuvre de l'apprentissage automatique dans l'entreprise. Cela s'explique par le fait que la collecte de ces données peut être coûteuse et prendre du temps dans de nombreux cas.

Data augmentation est un ensemble de techniques visant à augmenter artificiellement la quantité de données en générant de nouveaux points de données à partir de données existantes. Il s'agit notamment d'apporter de petites modifications aux données ou d'utiliser des modèles d'apprentissage profond pour générer de nouveaux points de données. 
![image](https://research.aimultiple.com/wp-content/webp-express/webp-images/uploads/2021/04/dataaugmentaion_simpleimage_stanford-1160x418.png.webp)
Les avantages de l'augmentation des données sont les suivants :
- Améliorer la précision des prédictions du modèle
    + ajouter plus de données d'entraînement dans les modèles
    + en réduisant du data overfitting et créer une variabilité dans les données
    + aider à résoudre les problèmes de déséquilibre des classes
    + ...
- Réduire les coûts de collecte et d'annotation des données
- Permet la prédiction d'événements rares

## Dataset Preprocessing
Après avoir téléchargé le jeu de données facemask de Kaggle et l'ajout des autres images à notre dataset, nous devons convertir les données d'image et leurs annotations dans un format de fichier TFRecord qui est pris en charge par TensorFlow.

#### Qu'est-ce que le TFRecord ?
La [documentation de TensorFlow](https://www.tensorflow.org/tutorials/load_data/tfrecord) décrit TFRecords comme un format simple pour stocker une séquence d'enregistrements binaires. Les données converties dans ce format occupent moins d'espace disque, ce qui rend chaque opération effectuée sur l'ensemble de données plus rapide. Il est également optimisé pour traiter les grands ensembles de données en les divisant en plusieurs sous-ensembles.

Le fichier TFRecord n'est rien d'autre qu'une liste de tf.train.Example qui est créée pour chaque image et son annotation. Le tf.train.Example est un dictionnaire {key : value} contenant, par exemple, le tableau d'octets des données d'image, les coordonnées des boundary boxes, les ids de classe et d'autres caractéristiques nécessaires.
La "key" est une chaîne de caractères contenant le nom de la caractéristique particulière et la "value" est la caractéristique elle-même qui doit être un des types tf.train.Feature (tf.train.BytesList, tf.train.FloatList ou tf.train.Int64List).

Mais avant de créer le fichier TFRecord on doit créer le fichier label_map.pbtxt, qui contient la correspondance entre le nom de la classe et l'identifiant de la classe. Label map est un autre fichier essentiel à l'entrainement du modèle. Il s'agit d'un simple fichier texte contenant une correspondance entre l'identifiant de la classe et le nom de la classe.
```
item { 
  id: 1
  name: "without_mask"
}

item { 
  id: 2
  name: "with_mask"
}

item { 
  id: 3
  name: "mask_weared_incorrect"
}
```
Voila la fonction qui génére le fichier label_map.pbtxt
[![image](https://www.linkpicture.com/q/labelmap.jpg)](https://www.linkpicture.com/view.php?img=LPic62616050e60bc1068159940)

Il nous faut maintenant une fonction qui puisse fusionner tous les fichiers PascalVOC .XML en un seul fichier CSV pour les données d'entraînement et de test.
Et voila la fonction qui cree le fichier csv, cette fonction convert() retournera 2 nouveaux fichiers sous Tensorflow/workspace/annotations nommés testlabels.csv et trainlabels.csv
[![image](https://www.linkpicture.com/q/xmltocsv.jpg)](https://www.linkpicture.com/view.php?img=LPic626160d529530467712458)

Il est maintenant temps de créer l'enregistrement TF (Tensorflow Record) utilisé pour entraîner le modèle.

Pour ce faire, j'ai utilisé les commandes :
```
# Train Record
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/train -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/train.record

# Test Record
python Tensorflow/scripts/generate_tfrecord.py -x Tensorflow/workspace/images/test -l Tensorflow/workspace/annotations/label_map.pbtxt -o Tensorflow/workspace/annotations/test.record
```
### Model Selection

Tout d'abord, nous devons choisir un modèle d'architecture avec lequel travailler. Heureusement, il y a beaucoup d'options et elles sont toutes géniales.
Nous pouvons nous rendre dans le  TFOD Model Zoo et choisir un modèle que nous allons essayer d'entraîner. Le  TFOD Model Zoo contient de nombreux modèles qui sont déjà pré-entraînés sur le jeu de données COCO 2017. Les modèles pré-entraînés nous permettent d'économiser beaucoup de temps d'apprentissage puisqu'ils sont déjà capables de détecter 90 catégories.
![image](https://i0.wp.com/neptune.ai/wp-content/uploads/models-tf2-model-zoo.png?resize=864%2C1024&ssl=1)
Pour ce projet, j'ai téléchargé l'un des modèles les plus petits et les plus rapides disponibles SSD MobileNet V2 FPNLite 320x320.
### Model Configuration :
Pour configurer ce modèle pré-entraîné pour notre utilisation, on crée un sous-dossier avec le nom du modèle (my_ssd_mobnet) que nous utilisons dans le dossier models.

[![image](https://www.linkpicture.com/q/model.jpg)](https://www.linkpicture.com/view.php?img=LPic62616d7519f9b1833022914)

On copie le fichier pipeline.config du dossier téléchargé dans le dossier my_ssd_mobnet. Ce fichier contient les configurations pour le pipeline de l'entrainement.
Nous devons modifier ces configurations pour répondre à nos besoins.
- num_classes : Nombre d'étiquettes de classe détectées à l'aide du détecteur, 3 labels pour mon cas. 
- batch_size : Batch size utilisé pour le training. On modifie cette valeur en fonction de la mémoire dont vous disposez. Une taille de lot plus élevée nécessite une mémoire plus importante.
- fine_tune_checkpoint : Chemin vers checkpoint du modèle pré-entraîné.
- fine_tune_checkpoint_type : "detection" pour mon cas.
- label_map_path : Chemin vers le fichier label_map_path que nous avons créé plus tôt.
- input_path : Chemin vers les fichiers tfrecord que nous avons créé plus tôt.
-  data_augmentation_options : pour mon cas j'ai utilisé différentes techniques d'augmentation par exemple (random_horizontal_flip, random_vertical_flip, random_crop_image ..)

Un autre paramètre couramment modifié est **learning_rate**, lorsque l'apprentissage est trop lent, nous pouvons augmenter le taux d'apprentissage, ou lorsque la perte d'apprentissage est trop instable, nous pouvons le diminuer.

## Entrainement et Evaluation
#### Training
Maintenant que nous avons prétraité notre dataset et configuré notre pipeline de l'entrainement, passons à l'entrainement de notre modèle.
Nous sommes maintenant prêts à entraîner notre modèle ; tout ce que nous avons à faire est d'exécuter la commande.
```
!python Tensorflow/tensorflow-models/models/research/object_detection/model_main_tf2.py --model_dir=Tensorflow/workspace/models/my_ssd_mobnet --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --num_train_steps=40000
```
- model_dir — avec l'emplacement du dossier my_ssd_mobnet.
- pipeline_config_path — avec l'emplacement du fichier pipeline.config dans le dossier my_ssd_mobnet.

Le modèle devrait commencer à s'entraîner sur la commande ci-dessus.Pendant qu'il s'entraîne, il évalue le modèle sur la base des derniers fichiers checkpoint en my_ssd_mobnet et les résultats sont enregistrés comme des fichiers events.out.tfevents.* en my_ssd_mobnet/train, Ces fichiers peuvent être utilisés pour monitoring les performances du modèle dans Tensorboard, comme indiqué à l'étape suivante.

#### Evaluation
Tensorboard est une fonctionnalité de Tensorflow qui nous permet de faire le monitoring de la performance de notre modèle, Ceci peut être utilisé pour analyser si le modèle est surajusté ou sous-ajusté ou s'il apprend quelque chose.
Nous pouvons utiliser les fichiers events.out.tfevents.* générés dans my_ssd_mobnet/train avec Tensorboard pour surveiller les performances de notre modèle.
Pour lancer Tensorboard, j'ai exécuté la commande suivante depuis le dossier du projet.

```
%load_ext tensorboard
%tensorboard --logdir=Tensorflow/workspace/models/my_ssd_mobnet/train
```
[![image](https://www.linkpicture.com/q/tensorboard.jpg)](https://www.linkpicture.com/view.php?img=LPic626274a1b7294615572798)
**** 

### Exporter le modele
Maintenant que nous avons entrainé nos données pour répondre à nos besoins, nous devons exporter ce modèle pour l'utiliser dans nos applications souhaitées.
Dans le dossier des scripts, nous trouverons le fichier exporter_main_v2.py. Ce code sera utilisé pour exporter notre modèle.

```
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config --trained_checkpoint_dir Tensorflow/workspace/models/my_ssd_mobnet --output_directory version1
```

### Predictions en Real Time  :
Maintenant c'est le temps de commencer à faire des détections. Nous devons d'abord charger notre modèle comme suit :
```
def load_model():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(
        model_config=configs['model'], is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-31')).expect_partial()
    return detection_model
```

##### Detection Function
Dans cette partie, nous créons une fonction qui prend une image, charge le modèle, prédit ce qui est dans l'image en analysant notre réseau neuronal et en renvoyant une classe 1, 2 ou 3, où 1 est "without_mask", 2 est "with_mask" et 3 est "mask_weared_incorrect".
```
def detect(image):
    detection_model = load_model()
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections
```
Maintenat on va créer une fonction qui convertit notre image d'entrée en un tenseur, c'est-à-dire une matrice composée des pixels de notre image qui, une fois introduite dans le modèle, sera manipulée par l'une des couches de convolutions au moyen d'opérations mathématiques complexes et renvoyée comme entrée pour la couche suivante jusqu'à ce qu'elle atteigne la dernière couche qui renverrait une classe. Si la classe est 0, alors l'objet est inconnu.
![image](https://miro.medium.com/max/1400/0*LsS61nqFVN2GiF00.png)
Avec cette fonction, nous pouvons ensuite dessiner des labels et des bounding boxes basées sur les coordonnées renvoyées par la fonction et utiliser OpenCV pour lire l'image et afficher la sortie avec les détections effectuées.
```
def check(input):
    TEST_IMAGE_PATHS = glob.glob(input)
    try:
        images = random.sample(TEST_IMAGE_PATHS, k=5)
    except:
        images = TEST_IMAGE_PATHS

    for image_path in images:
        print(image_path)
        image_np = load_image_into_numpy_array(image_path)
            
        input_tensor = tf.convert_to_tensor(
                np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        category_index = label_map_util.create_category_index_from_labelmap(
            ANNOTATION_PATH+'/label_map.pbtxt')
        
        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'][0].numpy(),
                    (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
                    detections['detection_scores'][0].numpy(),
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=100,
                    min_score_thresh=.7,
                    agnostic_mode=False,
        )
```
##### *Résultats* :
[![image](https://www.linkpicture.com/q/resultat.jpg)](https://www.linkpicture.com/view.php?img=LPic62627f5471a791579685319)
##### Real-Time Prediction

##### Déploiement : 
Cette partie se concentrera sur la façon dont nous pouvons servir les modèles de détection d'objets spécifiquement avec TF Serving. Comment créer des modèles de détection d'objets prêts pour la production, créer un environnement de service TF à l'aide de Docker, servir le modèle et créer un script côté client pour accéder au modèle.

##### Créer un modèle prêt à la production pour TF-Serving
Après avoir entraîné notre modèle de détection d'objets à l'aide de TensorFlow, nous aurons les quatre fichiers suivants enregistrés sur notre disque, 
![image](https://miro.medium.com/max/1052/1*wwIWGlWy5xfS54Woy9raag.png)
Pour créer des modèles prêts à être servis, on va utiliser le fichier exporter.py disponible sur Github de l'API de détection d'objets.
[![image](https://www.linkpicture.com/q/export_4.jpg)](https://www.linkpicture.com/view.php?img=LPic626677d4382f31897453895)
Explication du code ci-dessus,
- Chaque modèle de détection d'objets possède une configuration qui doit être transmise au fichier export_model.py. Elle consiste en des informations concernant l'architecture du modèle.
- La méthode get_configs_from_pipeline_file crée un dictionnaire à partir du fichier de configuration et la méthode create_pipeline_proto_from_configs crée un objet tampon buffer à partir de ce dictionnaire.
- input_checkpoint est le chemin vers model.ckpt du modèle entraîné.
- model_version_id est un entier pour la version actuelle du modèle. Ceci est requis par TF-serving pour le versioning des modèles.
- object_detection.exporter sauvegardera le modèle dans le format suivant,

1 est la version du modèle, saved_model.pb contient l'architecture du modèle et le répertoire variables contient les poids du modèle. Ce modèle est prêt à être servi.

![image](https://miro.medium.com/max/1324/1*2jVWPNBi25L8hTmt0tal7w.png)

### Créer un environnement de TF-serving à l'aide de Docker


