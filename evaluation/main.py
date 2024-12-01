"""This code is adopted from RDED here: https://github.com/LINs-lab/RDED/tree/main"""
from argument import args
from validation.main import (
    main as valid_main,
)  # The relabel and validation are combined here for fast experiment
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    valid_main(args)
