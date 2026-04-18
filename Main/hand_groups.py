from enum import Enum

class HandGroup(Enum):
    SYM3 = "sym3"     # symmetric, 3 fingers
    SYM4 = "sym4"     # symmetric, 4 fingers
    SYM5 = "sym5"     # symmetric, 5 fingers
    ANTH21 = "anth21" # anthro, thumb slot 21
    ANTH27 = "anth27" # anthro, thumb slot 27
    ANTH33 = "anth33" # anthro, thumb slot 33