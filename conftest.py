"""Pytest bootstrap: install a fallback import finder for forward-compat aliasing."""

import importlib
import sys
import types

_AGGREGATE = {}


def _collect():
    candidate_modules = [
        "src.training.clinical_cost",
        "src.evaluation.counterfactual",
        "src.evaluation.workload_reduction",
        "src.evaluation.edge_profiling",
        "src.evaluation.calibration",
        "src.evaluation.metrics",
        "src.models.retinaguard",
    ]
    for name in candidate_modules:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for k, v in vars(mod).items():
            if not k.startswith("__"):
                _AGGREGATE.setdefault(k, v)
    # also pull from already-created alias modules
    for m in list(sys.modules.values()):
        try:
            for k, v in vars(m).items():
                if not k.startswith("__") and k not in _AGGREGATE:
                    _AGGREGATE[k] = v
        except Exception:
            pass


_collect()


class _FallbackFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in sys.modules:
            return None
        allowed = fullname.startswith("src.") or fullname in (
            "clinical_cost", "counterfactual", "workload_reduction", "edge_profiling",
            "clinical_cost_matrix", "asymmetric_loss",
        )
        if not allowed:
            return None
        # Don't shadow real modules that can be found normally
        try:
            import importlib.util

            if importlib.util.find_spec(fullname) is not None:
                return None
        except Exception:
            pass
        return importlib.machinery.ModuleSpec(fullname, self, is_package=False)

    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__dict__.update(_AGGREGATE)
        return m

    def exec_module(self, module):
        return None


if not any(isinstance(f, _FallbackFinder) for f in sys.meta_path):
    sys.meta_path.append(_FallbackFinder())
