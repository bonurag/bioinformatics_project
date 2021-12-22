# ===============
# DIRECTORY
# ===============

TUNER_DIR = "tuners/tuner_hp_"

MODEL_TYPE_FFNN = "ffnn"
MODEL_TYPE_CNN = "cnn"
MODEL_TYPE_MMNN = "mmnn"

TUNER_DIR_FFNN = TUNER_DIR+MODEL_TYPE_FFNN
TUNER_DIR_CNN = TUNER_DIR+MODEL_TYPE_CNN
TUNER_DIR_MMNN = TUNER_DIR+MODEL_TYPE_MMNN

TUNER_PROJECT_NAME_FFNN = "bioinformatics_project_ffnn_hp"
TUNER_PROJECT_NAME_CNN = "bioinformatics_project_cnn_hp"
TUNER_PROJECT_NAME_MMNN = "bioinformatics_project_mmnn_hp"

MODELS_TYPE = [MODEL_TYPE_FFNN, MODEL_TYPE_CNN, MODEL_TYPE_MMNN]

HP_MAX_EPOCHS_FFNN = 30
HP_MAX_EPOCHS_CNN = 35
HP_MAX_EPOCHS_MMNN = 25

# ===============
# DATA
# ===============

HOLDOUTS_NUM_SPLIT = 10
TEST_SIZE = 0.2

# ===============
# EPIGENOMIC DATA
# ===============

WINDOW_SIZE = 256
CELL_LINE = "H1"
GENOME_CACHE_DIR = "/bio_data/genomes"

# ===============
# FFNN
# ===============

FFNN_NAME_HP = "ffnn_hp"
FFNN_NAME = "BinaryClassificationFFNN"

# ===============
# CNN
# ===============

CNN_NAME_HP = "ffnn_hp"
CNN_NAME = "BinaryClassificationCNN"

# ===============
# MMNN
# ===============

MMNN_NAME_HP = "mmnn_hp"
MMNN_SIMPLE = "MMNN"
MMNN_BOOST = "BoostedMMNN"
