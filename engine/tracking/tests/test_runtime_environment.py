from run_pipeline import normalize_thread_environment


def test_invalid_autodl_thread_count_is_normalized():
    environment = {"OMP_NUM_THREADS": "0", "MKL_NUM_THREADS": "bad", "OPENBLAS_NUM_THREADS": "8"}
    normalize_thread_environment(environment)
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "8"
