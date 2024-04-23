# README OTZETA
He subido todo el código por si acaso, para que luego no haya alguna dependencia que se me haya escapado. 

Sin embargo, no todos los archivos son importantes. A cada archivo importante le pondré una sección en el readme. Actualmente, he dejado el modelo a un lado porque no estaba obteniendo buenos resultados, y me he puesto a trabajar con la conversión quaternion - Euler. Aun así, lo importante es toda la parte del modelo.

IMPORTANTE: en sí, sólo interesan las carpetas: rnn y util. Todo lo demás son carpetas de todas las cosas que he ido probando, pero no sirven para nada ahora mismo.
## rnn/lstm.py
Es la clase que contiene el modelo y su entrenamiento.

Para el entrenamiento, primero creo un StandardScaler del paquete skLearn, cargando todos los datos juntos (no creo secuencias, simplemente cargo todos los vectores juntos).

    prepareScalerForJump(0, "", "dataset")

Una vez que se haya creado el scaler con los datos, se puede comentar esta línea, porque lo guardo como pickle, y luego lo cargo cada vez que lo necesite.

El segundo paso es cargar todas las secuencias (el dataset). Cargo dos instancias de lstmDataset para ello, uno de train y otro de validation

    datamodule = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Train", datasetName = "dataset", sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = useDifferences, jump = jump)

    datamoduleVal = lstmDataset(root="/home/bee/Desktop/idle animation generator", batchSize = 128, partition="Validation", datasetName = "dataset", sequenceSize = n_steps, trim=False, verbose=True, outSequenceSize=n_steps_out, removeHandsAndFace = True, scaler = scaler, loadDifferences = useDifferences, jump = jump)

Después creo y entreno el modelo.

    model = Sequential()
    model.add(LSTM(1000, activation = 'tanh', input_shape=(n_steps, n_features), return_sequences = True, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(LSTM(1000, activation = 'tanh', return_sequences = False, dropout = 0.4, recurrent_dropout = 0.4))
    model.add(Dense(n_features))
    opt = Adam(learning_rate=learning_rate, clipnorm = 0.01)
    # opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mae')
    model.summary()
    history = model.fit(datamodule, validation_data=datamoduleVal, epochs=500, verbose=1)
    
    model.save("models/differencesDataset" + str(jump) + ".keras")

## rnn/lstmDataset.py
Esta clase es la que carga los datasets, y prepara automáticamente los batches y todo para el entrenamiento. La clase es muy simple, porque es sólo un wrapper para cargar los datos. Hace uso de util/bvhLoader para ello.

Lo único complejo de la clase es el método __getitem__ (y puede que esto esté mal, porque cuando lo programé lo comprobé, pero es un método muy complejo y puede que la liase).

Lo que hace es: para no tener que cargar una cantidad enorme de secuencias, cargo en el dataset secuencias de animación enteras, y luego éste método devuelve slices de una secuencia más larga. Por ejemplo imaginemos que en el dataset hay una secuencia de 1234 frames, y yo estoy entrenando el modelo con secuencias de tamaño 500 de input y 100 de output.

Este método, al cargar el batch número 34 (por ejemplo), carga una instancia específica: podría ser, por ejemplo, la instancia número 3678: la secuencia que empieza en el frame 165 de la persona número 3. Por lo tanto, el método calcula primero cual es el número de la secuencia (3) y el frame de comienzo (56), y cargaría desde ese frame, hasta el frame 556 como input, y desde 557 hasta 657 como output.

En teoría no debería cargar nunca una secuencia de dos personas distintas. Es decir, tomando el ejemplo anterior, no llegaría nunca al frame 1234 de la persona 3 y no saltaría a la persona 4.

Este método fue el más problemático de programar para mí, y se me hace bastante difícil debuggearlo. Sin embargo, creo que funciona bien en sí, por las pruebas que fui haciendo al programarlo.
## util/bvhLoader.py
Esta clase tiene todos los métodos para cargar los datos de distintas maneras. Esta última clase también tiene lo suyo. En sí tiene métodos bastante simples, pero son muchos, porque durante el desarrollo he estado probando muchísimas cosas y he necesitado muchos métodos distintos para cargar datos.

Primero de todo, para usar esta clase en las otras clases, lo he tenido que importar de esta manera, porque no podía hacer un import normal:

    import sys
    sys.path.insert(1, '/home/bee/Desktop/idle animation generator/codeNew/util')
    import bvhLoader

No sé por qué pasa esto. Puede que tengas que cambiar el path de la carpeta /util en tu PC.

Lo segundo, también puede que tengas que cambiar los paths para cargar los datasets, en la línea 132 y en la 205:

    if partition=="All":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/"
    if partition=="Train":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_trn"
    if partition=="Validation":
        path = "/home/bee/Desktop/idle animation generator/" + datasetName + "/genea2023_val"

Después, los métodos importantes son:

    def loadBvhToList(path, returnHeader = False, returnData = True, returnCounter = True, 
                  onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, jump = 0, useQuaternions=False):

Carga un único archivo BVH y devuelve una lista de vectores. Puedes elegir:
* Si devolver el header del archivo, si devolver los datos, y si devolver el número de frames cargados.
* Si cargar solo posiciones o rotaciones (no lo uso, y no funciona bien).
* Si quitar la cara y manos (solo para el dataset silenceDataset3sec, y no funciona bien).
* Si cargar todos los frames o hacer saltos y cargar uno de cada X frames (jump).

El siguiente método carga un dataset entero. Los parámetros son casi iguales porque solo los redirige para usarlos en el método __loadBvhToList__.

    def loadDataset(datasetName, partition = "All", specificSize=-1, verbose = False, trim = False, specificTrim = -1, 
                onlyPositions = False, onlyRotations = False, removeHandsAndFace = False, scaler = None, 
                loadDifferences = False, jump = 0, useQuaternions = False)

Este método carga los datos, y devuelve una lista de listas (una lista que contiene secuencias/animaciones uno a uno con todos sus frames). Puedes elegir:

* Cuantas secuencias cargar (specificSize)
* Cuantos frames puede tener como máximo cada secuencia (specificTrim)
* Escalar los datos o no (pasándole un objeto scaler)
* Cargar las diferencias entre frames o frames normales (loadDifferences)

Por otra parte hay métodos para cargar un dataset entero, pero sin hacer secuencias, es decir cargar todos los vectores a la vez (se usa para crear el standarScaler).

    def loadDatasetInBulk(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace=False, jump=0):

Para crear los propios standardScaler:

    def createAndFitStandardScaler(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace = False):

Y: 

    def createAndFitStandardScalerForDifferences(datasetName, partition = "All", specificSize=-1, verbose = False, removeHandsAndFace = False, jump=0, onlyPositions=False, onlyRotations=False):

Para cargar el dataset en forma de secuencias (esto ya no se usa, porque creo las secuencias en el runtime). Este método debería de estar comentado, pero no lo he hecho, a ver si se rompe algo que no debería

    def loadSequenceDataset(datasetName, partition = "All", specificSize = -1, verbose = False, sequenceSize = 10, trim = False, specificTrim = -1, onlyPositions = False, onlyRotations = False, outSequenceSize=1, removeHandsAndFace = False, scaler = None, loadDifferences = False, jump = 0)