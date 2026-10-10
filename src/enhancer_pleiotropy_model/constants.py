"""Fixed geometry shared by data preparation, models and training."""

ASSAYS = ("atac", "h3k27ac")
# Context order of every label, profile and model output.
CONTEXTS = ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid")
CONTEXT_NAMES = {
    "ab": "adult brain", "e13": "embryo stage 13", "e5": "embryo stage 5",
    "ead": "eye-antennal disc", "hid": "haltere disc", "lb": "larval brain",
    "o": "ovary", "wid": "wing disc",
}
SPLITS = ("train", "validation", "test")

INPUT_BP = 2048
ATAC_TARGET_BP = 512          # central ATAC profile
H3K27AC_TARGET_BP = 1536      # central H3K27ac profile
SOURCE_BIN_BP = 16            # profile resolution on disk
H3K27AC_OUTPUT_POOL_SIZE = 4  # H3K27ac is predicted at 64 bp
ATAC_BINS = ATAC_TARGET_BP // SOURCE_BIN_BP                                   # 32
H3K27AC_BINS = H3K27AC_TARGET_BP // SOURCE_BIN_BP // H3K27AC_OUTPUT_POOL_SIZE  # 24
