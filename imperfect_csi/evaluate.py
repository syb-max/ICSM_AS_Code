"""Compatibility entry point for the canonical independent paper evaluator."""
from icsm_as.evaluation.imperfect_csi import (
    ChannelBank, checked_subset, error_db, evaluate_case, gram, summarize,
    synthetic_estimate, paired_cluster_ci, check_paper_dimensions,
    check_paper_test_rows, measure_selector, main, parser,
)

if __name__ == "__main__":
    main(parser().parse_args())
