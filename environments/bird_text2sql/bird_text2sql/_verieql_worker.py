from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path


def _patch_z3_context_keywords() -> None:
    """Bridge VeriEQL's ctx-aware wrappers to stock z3py releases.

    VeriEQL vendors optimized z3py files whose And/Or/Sum functions accept a
    ``ctx`` keyword.  The binary ``z3-solver`` wheel does not expose that
    keyword for those three helpers.  Non-empty expressions infer their
    context from their arguments; only the identity values need it explicitly.
    """
    import z3

    def values(args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, set)):
            return tuple(args[0])
        return args

    stock_and, stock_or, stock_sum = z3.And, z3.Or, z3.Sum

    def compatible_and(*args, ctx=None):
        flattened = values(args)
        return stock_and(*flattened) if flattened else z3.BoolVal(True, ctx=ctx)

    def compatible_or(*args, ctx=None):
        flattened = values(args)
        return stock_or(*flattened) if flattened else z3.BoolVal(False, ctx=ctx)

    def compatible_sum(*args, ctx=None):
        flattened = values(args)
        return stock_sum(*flattened) if flattened else z3.IntVal(0, ctx=ctx)

    z3.And = compatible_and
    z3.Or = compatible_or
    z3.Sum = compatible_sum


def _make_importable() -> None:
    override = os.environ.get("VERIEQL_PATH")
    if override:
        sys.path.insert(0, str(Path(override).resolve()))
    try:
        import environment  # noqa: F401

        return
    except ImportError:
        pass
    try:
        distribution = importlib.metadata.distribution("verieql")
    except importlib.metadata.PackageNotFoundError:
        return
    for item in distribution.files or ():
        if item.name == "environment.py":
            sys.path.insert(0, str(Path(item.locate()).resolve().parent))
            return
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        source = json.loads(direct_url).get("url", "")
        if source.startswith("file://"):
            sys.path.insert(0, source.removeprefix("file://"))


def main() -> None:
    request = json.loads(sys.stdin.read())
    _patch_z3_context_keywords()
    _make_importable()
    try:
        import environment as environment_module

        os.chdir(Path(environment_module.__file__).resolve().parent)
        from environment import Environment
    except Exception as exc:
        print(json.dumps({"result": None, "error": f"verieql_unavailable: {type(exc).__name__}: {exc}"}))
        return
    if request.get("probe") is True:
        print(json.dumps({"result": True}))
        return
    try:
        with Environment(generate_code=False, timer=False, show_counterexample=False) as verifier:
            for table, columns in request["schema"].items():
                verifier.create_database(
                    attributes=columns,
                    bound_size=int(request.get("bound_size", 2)),
                    name=table,
                )
            verifier.save_checkpoints()
            result = verifier.analyze(request["predicted_sql"], request["gold_sql"])
        print(json.dumps({"result": result if isinstance(result, bool) else False}))
    except Exception as exc:
        print(json.dumps({"result": None, "error": f"{type(exc).__name__}: {exc}"}))


if __name__ == "__main__":
    main()
