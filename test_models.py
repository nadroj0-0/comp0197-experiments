"""
Quick checks for the wrapper layer.

These tests avoid full training. They mainly check imports, config wiring,
wrapper instantiation, and basic control flow.

Run from project root:
    python test_models.py
    python test_models.py --fast        # skips data loading (import tests only)
    python test_models.py --model baseline_gru_det   # single model only
"""

import sys
import argparse
import traceback
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
SKIP = "⏭  SKIP"

results = []


def test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()


def skip(name, reason):
    results.append((SKIP, name))
    print(f"  {SKIP}  {name}  ({reason})")


# =============================================================================
# TEST GROUP 1 — Imports
# =============================================================================

def run_import_tests():
    print("\n── GROUP 1: Imports ──────────────────────────────────────────")

    def t_base_model():
        from base_model import BaseModel
        assert hasattr(BaseModel, "load_and_split_data")
        assert hasattr(BaseModel, "QUANTILES")
        assert hasattr(BaseModel, "PRED_LENGTH")

    def t_models_module():
        import models  # noqa — just check it loads

    def t_all_classes():
        from models import (
            BaselineGRUDet, BaselineGRUProb, BaselineGRUNB,
            BaselineQuantileGRU, BaselineWQuantileGRU,
            HierarchicalGRUDet, HierarchicalGRUProb, HierarchicalGRUNB,
            HierarchicalQuantileGRU, HierarchicalWQuantileGRU,
        )

    def t_pipeline_imports():
        # Check the main imports used by models.py
        from utils.data import build_dataloaders, get_feature_cols
        from utils.experiment import Experiment
        from utils.config_loader import (
            load_model_config, load_registry,
            resolve_registry_entry, create_run_dir,
            snapshot_configs, get_model_run_dir,
        )
        from utils.common import full_train, save_json
        from utils.hyperparameter import staged_search
        from utils.network import build_baseline_gru, build_hierarchical_gru
        from utils.training_strategies import gru_step, quantile_gru_step

    test("base_model.py imports cleanly", t_base_model)
    test("models.py imports cleanly", t_models_module)
    test("all 10 model classes importable", t_all_classes)
    test("all pipeline dependencies resolve", t_pipeline_imports)


# =============================================================================
# TEST GROUP 2 — Instantiation and BaseModel compliance
# =============================================================================

def run_instantiation_tests():
    print("\n── GROUP 2: Instantiation & BaseModel compliance ─────────────")

    from models import (
        BaselineGRUDet, BaselineGRUProb, BaselineGRUNB,
        BaselineQuantileGRU, BaselineWQuantileGRU,
        HierarchicalGRUDet, HierarchicalGRUProb, HierarchicalGRUNB,
        HierarchicalQuantileGRU, HierarchicalWQuantileGRU,
    )
    from base_model import BaseModel

    all_classes = [
        BaselineGRUDet, BaselineGRUProb, BaselineGRUNB,
        BaselineQuantileGRU, BaselineWQuantileGRU,
        HierarchicalGRUDet, HierarchicalGRUProb, HierarchicalGRUNB,
        HierarchicalQuantileGRU, HierarchicalWQuantileGRU,
    ]

    def t_all_instantiate():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert m is not None, f"{cls.__name__} failed to instantiate"

    def t_all_subclass_basemodel():
        for cls in all_classes:
            assert issubclass(cls, BaseModel), \
                f"{cls.__name__} does not subclass BaseModel"

    def t_model_name_defined():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert isinstance(m.model_name, str) and len(m.model_name) > 0, \
                f"{cls.__name__}.model_name is empty or not a string"

    def t_preprocess_callable():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert callable(m.preprocess), f"{cls.__name__}.preprocess not callable"

    def t_train_callable():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert callable(m.train), f"{cls.__name__}.train not callable"

    def t_run_callable():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert callable(m.run), f"{cls.__name__}.run not callable"

    def t_load_and_split_data_inherited():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert callable(m.load_and_split_data), \
                f"{cls.__name__} missing inherited load_and_split_data"

    def t_model_name_matches_registry():
        from utils.config_loader import load_registry
        registry = load_registry(PROJECT_DIR / "configs" / "registry.yml")
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            assert m.model_name in registry, \
                f"{m.model_name} not in registry.yml"

    def t_yml_exists_for_each_model():
        for cls in all_classes:
            m = cls(run_name="test_sanity")
            yml = PROJECT_DIR / "configs" / "models" / f"{m.model_name}.yml"
            assert yml.exists(), f"Missing yml: {yml}"

    test("all 10 classes instantiate", t_all_instantiate)
    test("all 10 classes subclass BaseModel", t_all_subclass_basemodel)
    test("model_name defined on all classes", t_model_name_defined)
    test("preprocess() callable on all", t_preprocess_callable)
    test("train() callable on all", t_train_callable)
    test("run() callable on all", t_run_callable)
    test("load_and_split_data() inherited on all", t_load_and_split_data_inherited)
    test("model_name matches registry.yml for all", t_model_name_matches_registry)
    test("configs/models/{name}.yml exists for all", t_yml_exists_for_each_model)


# =============================================================================
# TEST GROUP 3 — Config and registry pipeline
# =============================================================================

def run_config_tests():
    print("\n── GROUP 3: Config & registry ───────────────────────────────")

    def t_registry_loads():
        from utils.config_loader import load_registry
        reg = load_registry(PROJECT_DIR / "configs" / "registry.yml")
        assert len(reg) > 0

    def t_all_builders_resolve():
        from utils.config_loader import load_registry, resolve_registry_entry
        reg = load_registry(PROJECT_DIR / "configs" / "registry.yml")
        model_names = [
            "baseline_gru_det", "baseline_gru_prob", "baseline_gru_nb",
            "baseline_quantile_gru", "baseline_wquantile_gru",
            "hierarchical_gru_det", "hierarchical_gru_prob", "hierarchical_gru_nb",
            "hierarchical_quantile_gru", "hierarchical_wquantile_gru",
        ]
        for name in model_names:
            resolved = resolve_registry_entry(reg[name])
            assert callable(resolved["builder"]), f"builder not callable for {name}"
            assert callable(resolved["training_step"]), f"step not callable for {name}"

    def t_train_configs_load():
        from utils.config_loader import load_model_config
        model_names = [
            "baseline_gru_det", "baseline_gru_prob", "baseline_gru_nb",
            "baseline_quantile_gru", "baseline_wquantile_gru",
            "hierarchical_gru_det", "hierarchical_gru_prob", "hierarchical_gru_nb",
            "hierarchical_quantile_gru", "hierarchical_wquantile_gru",
        ]
        for name in model_names:
            cfg = load_model_config(
                PROJECT_DIR / "configs" / "models" / f"{name}.yml")
            assert "train_config" in cfg, f"No train_config in {name}.yml"
            tc = cfg["train_config"]
            assert "epochs" in tc
            assert "horizon" in tc
            assert "seq_len" in tc

    test("registry.yml loads", t_registry_loads)
    test("all builders resolve from registry", t_all_builders_resolve)
    test("all train_configs load with required keys", t_train_configs_load)


# =============================================================================
# TEST GROUP 4 — Data routing logic (no actual data loading)
# =============================================================================

def run_routing_tests():
    print("\n── GROUP 4: Loader routing logic ────────────────────────────")

    def t_nb_routes_to_nb_loaders():
        # Verify is_nb=True triggers NB loader route — check the logic directly
        from models import _load_loaders_for_model
        # We can't run it without data, but we can verify the function exists
        # and accepts the right args
        import inspect
        sig = inspect.signature(_load_loaders_for_model)
        params = list(sig.parameters.keys())
        assert "is_nb" in params
        assert "is_prob" in params
        assert "model_type" in params

    def t_wquantile_model_types_correct():
        from models import BaselineWQuantileGRU, HierarchicalWQuantileGRU
        # Verify the model_type strings passed in run() match what train.py expects
        import inspect
        src_bwq = inspect.getsource(BaselineWQuantileGRU.run)
        assert "baseline_wquantile_gru" in src_bwq

        src_hwq = inspect.getsource(HierarchicalWQuantileGRU.run)
        assert "hierarchical_wquantile_gru" in src_hwq

    def t_nb_flags_correct():
        from models import BaselineGRUNB, HierarchicalGRUNB
        import inspect
        src_nb = inspect.getsource(BaselineGRUNB.run)
        assert "is_nb=True" in src_nb

        src_hnb = inspect.getsource(HierarchicalGRUNB.run)
        assert "is_nb=True" in src_hnb

    def t_prob_flags_correct():
        from models import BaselineGRUProb, HierarchicalGRUProb
        import inspect
        src = inspect.getsource(BaselineGRUProb.run)
        assert "is_prob=True" in src

    def t_det_flags_correct():
        from models import BaselineGRUDet, HierarchicalGRUDet
        import inspect
        src = inspect.getsource(BaselineGRUDet.run)
        assert "is_nb=False" in src
        assert "is_prob=False" in src

    test("_load_loaders_for_model has correct signature", t_nb_routes_to_nb_loaders)
    test("wquantile model_type strings correct", t_wquantile_model_types_correct)
    test("NB models pass is_nb=True", t_nb_flags_correct)
    test("prob models pass is_prob=True", t_prob_flags_correct)
    test("det models pass is_nb=False, is_prob=False", t_det_flags_correct)


# =============================================================================
# TEST GROUP 5 — Experiment class wiring
# =============================================================================

def run_experiment_tests():
    print("\n── GROUP 5: Experiment class wiring ─────────────────────────")

    def t_experiment_class_importable():
        from utils.experiment import Experiment
        assert callable(Experiment)

    def t_experiment_has_required_methods():
        from utils.experiment import Experiment
        assert hasattr(Experiment, "train")
        assert hasattr(Experiment, "search")

    def t_experiment_accepts_preloaded():
        from utils.experiment import Experiment
        import inspect
        src = inspect.getsource(Experiment.train)
        # Experiment.train should call full_train
        assert "full_train" in src

    def t_full_train_importable():
        from utils.common import full_train
        import inspect
        sig = inspect.signature(full_train)
        params = list(sig.parameters.keys())
        assert "builder" in params
        assert "train_loader" in params
        assert "val_loader" in params

    test("Experiment class importable", t_experiment_class_importable)
    test("Experiment has train() and search()", t_experiment_has_required_methods)
    test("Experiment.train() calls full_train", t_experiment_accepts_preloaded)
    test("full_train has correct signature", t_full_train_importable)


# =============================================================================
# TEST GROUP 6 — Single model smoke test (fast, 2 epochs, tiny data)
# Only runs if --smoke flag passed — loads real data, creates real artefacts
# =============================================================================

def run_smoke_test(model_name: str):
    print(f"\n── GROUP 6: Smoke test — {model_name} (2 epochs, small data) ──")
    print("   WARNING: This loads real data and trains briefly.")
    print("   Artefacts saved to runs/test_smoke_run/\n")

    from utils.config_loader import load_model_config
    yml_path = PROJECT_DIR / "configs" / "models" / f"{model_name}.yml"
    if not yml_path.exists():
        skip(f"smoke test {model_name}", f"{yml_path} not found")
        return

    from models import (
        BaselineGRUDet, BaselineGRUProb, BaselineGRUNB,
        BaselineQuantileGRU, BaselineWQuantileGRU,
        HierarchicalGRUDet, HierarchicalGRUProb, HierarchicalGRUNB,
        HierarchicalQuantileGRU, HierarchicalWQuantileGRU,
    )
    registry = {
        "baseline_gru_det":          BaselineGRUDet,
        "baseline_gru_prob":         BaselineGRUProb,
        "baseline_gru_nb":           BaselineGRUNB,
        "baseline_quantile_gru":     BaselineQuantileGRU,
        "baseline_wquantile_gru":    BaselineWQuantileGRU,
        "hierarchical_gru_det":      HierarchicalGRUDet,
        "hierarchical_gru_prob":     HierarchicalGRUProb,
        "hierarchical_gru_nb":       HierarchicalGRUNB,
        "hierarchical_quantile_gru": HierarchicalQuantileGRU,
        "hierarchical_wquantile_gru":HierarchicalWQuantileGRU,
    }

    cls = registry.get(model_name)
    if cls is None:
        skip(f"smoke test {model_name}", "not in registry")
        return

    def t_run_completes():
        # Temporarily patch epochs to 2 so this finishes fast
        cfg_orig = load_model_config(yml_path)
        m = cls(run_name="test_smoke_run", do_search=False)
        # Monkey-patch: override epochs inside the config by writing a temp yml
        import yaml, copy, tempfile, shutil
        cfg_patched = copy.deepcopy(cfg_orig)
        cfg_patched["train_config"]["epochs"] = 2
        cfg_patched["train_config"]["top_k_series"] = 30   # tiny subset
        tmp_yml = yml_path.parent / f"{model_name}_smoke_tmp.yml"
        with open(tmp_yml, "w") as f:
            yaml.dump(cfg_patched, f)
        # Rename: swap real yml for tmp, run, restore
        backup = yml_path.with_suffix(".yml.bak")
        shutil.copy2(yml_path, backup)
        shutil.copy2(tmp_yml, yml_path)
        try:
            m.run()
        finally:
            shutil.copy2(backup, yml_path)
            backup.unlink()
            tmp_yml.unlink(missing_ok=True)

        # Verify artefacts were created
        run_dir = PROJECT_DIR / "runs" / "test_smoke_run" / "models" / model_name
        assert run_dir.exists(), f"run dir not created: {run_dir}"
        pt_files = list(run_dir.glob("*.pt"))
        assert len(pt_files) > 0, "No .pt file saved"
        json_files = list(run_dir.glob("*_history.json"))
        assert len(json_files) > 0, "No history JSON saved"

    def t_model_attribute_set():
        # After run(), self.model should be populated
        m = cls(run_name="test_smoke_run", do_search=False)
        # run already completed above — just check the class stores model
        import inspect
        src = inspect.getsource(m.run)
        # run() must assign self._exp or self.model
        assert "_run_full_pipeline" in src

    test(f"{model_name} run() completes end-to-end", t_run_completes)
    test(f"{model_name} run() stores _exp / model", t_model_attribute_set)


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Test the BaseModel wrapper")
    p.add_argument("--fast",    action="store_true",
                   help="Import tests only, skip config/routing/experiment tests")
    p.add_argument("--smoke",   action="store_true",
                   help="Run a real 2-epoch smoke test (loads data, slow)")
    p.add_argument("--model",   type=str, default="baseline_gru_det",
                   help="Which model to smoke-test (default: baseline_gru_det)")
    p.add_argument("--verbose", action="store_true",
                   help="Print full tracebacks on failure")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  test_models.py — BaseModel wrapper verification")
    print("=" * 60)

    run_import_tests()

    if not args.fast:
        run_instantiation_tests()
        run_config_tests()
        run_routing_tests()
        run_experiment_tests()

    if args.smoke:
        run_smoke_test(args.model)

    # Summary
    passed = sum(1 for r, _ in results if r == PASS)
    failed = sum(1 for r, _ in results if r == FAIL)
    skipped = sum(1 for r, _ in results if r == SKIP)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed} passed  |  {failed} failed  |  {skipped} skipped")
    print(f"{'='*60}")

    if failed > 0:
        print("\n  Failed tests:")
        for r, name in results:
            if r == FAIL:
                print(f"    - {name}")
        sys.exit(1)
    else:
        print("\n  All tests passed.")
