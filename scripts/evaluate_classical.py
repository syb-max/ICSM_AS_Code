#!/usr/bin/env python3
"""Paper evaluation: classical and neural methods must share one imperfect-CSI run."""
from icsm_as.evaluation.imperfect_csi import main, parser

if __name__ == "__main__":
    main(parser().parse_args())
