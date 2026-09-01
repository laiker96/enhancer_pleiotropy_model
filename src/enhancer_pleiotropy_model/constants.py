"""Frozen production constants."""

ASSAYS = ("atac", "h3k27ac")
CONTEXTS = ("ab", "e13", "e5", "ead", "hid", "lb", "o", "wid")
DNA_ALPHABET = frozenset("ACGT")
COMPLEMENT_INDICES = (3, 2, 1, 0)

INPUT_BP = 2048
ATAC_TARGET_BP = 512
H3K27AC_TARGET_BP = 1536
SOURCE_BIN_BP = 16
H3K27AC_OUTPUT_POOL_SIZE = 4
