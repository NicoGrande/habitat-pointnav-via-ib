from yacs.config import CfgNode as CN

_C = CN()

_C.PTH_GPU_ID = 0

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DIR = "/srv/share2/samyak/pointnav-egomotion-data/data"
_C.DATA.DATASET = "gibson"
_C.DATA.TYPE = "random"
_C.DATA.SPLIT = "train"
_C.DATA.ITEMS_PER_BATCH = 64
_C.DATA.NUM_WORKERS = 0

# ------------------------------------------------------------
# Task
# ------------------------------------------------------------
_C.TASK = CN()
_C.TASK.NAME = "action"
_C.TASK.NUM_CLASSES = 3
_C.TASK.NUM_REGRESSION_TARGETS = 4

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.REPR_SIZE = 512
_C.MODEL.INPUT_MODALITY = "rgb,depth"
_C.MODEL.N_ACTIONS = 3
_C.MODEL.ACTION_EMBEDDING_SIZE = 32

# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------
_C.MAX_EPOCHS = 1
_C.START_EPOCH = 0
_C.START_ITER = 0
_C.LEARNING_RATE = 0.0002
_C.ACTION_LOSS_WEIGHT = 1.0
_C.EGOMOTION_LOSS_WEIGHT = 1.0

# ------------------------------------------------------------
# Bookkeeping
# ------------------------------------------------------------
_C.EVAL_EVERY = 800
_C.PRINT_EVERY = 100
_C.PLOT_EVERY = 10
_C.SEED = 1234

_C.ENV_NAME = "sensors=rgb+deptth_task=action"
_C.LOG_DIR = "standalone/data/logs"
_C.CHECKPOINTS_DIR = "standalone/data/checkpoints"
_C.TENSORBOARD_DIR = "standalone/data/plots"