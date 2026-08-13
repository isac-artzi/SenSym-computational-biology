"""Regenerate every figure in the project, in one command.

    python experiments/make_all_figures.py            # full parameters (minutes)
    python experiments/make_all_figures.py --smoke    # tiny smoke run (< 1 min)

Use --smoke to check that nothing is broken; use the full run for anything
that will be committed or cited. Every script is seeded, so a full run
reproduces exactly — that is the property the Week 14 freeze test checks.

Exit status is 0 only if every script ran without raising. A script whose
week has not been implemented yet raises NotImplementedError, which is
reported as SKIPPED rather than as a failure: early in the semester most of
this list is expected to skip.
"""

import argparse
import importlib
import sys
import time
import traceback

SCRIPTS = [
    "week01_counting_noise",
    "week02_thinning",
    "week03_dprime_depth",
    "week04_accuracy_roc",
    "week05_sqrt_k",
    "week06_bead_design",
    "week07_simulator",
    "week08_classifiers",
    "week09_dropout",
    "week10_markers",
    "week11_overdispersion",
    "week12_beads",
    "week13_error_bars",
    "week14_report_figures",
    "week15_poster_figures",
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny parameters — checks it runs, not that it is right")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these scripts (names without .py)")
    args = ap.parse_args(argv)

    names = args.only or SCRIPTS
    ok, skipped, failed, all_paths = [], [], [], []

    for name in names:
        print("\n" + "=" * 70)
        print(f"=== {name}" + ("  [smoke]" if args.smoke else ""))
        print("=" * 70)
        t0 = time.time()
        try:
            mod = importlib.import_module(name)
            paths = mod.main(fast=args.smoke) or []
            all_paths.extend(paths)
            ok.append((name, time.time() - t0, len(paths)))
        except NotImplementedError as exc:
            skipped.append((name, str(exc)[:70]))
            print(f"--- SKIPPED: not implemented yet ({exc})")
        except Exception:
            failed.append(name)
            traceback.print_exc()
            print(f"--- FAILED: {name}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, dt, n in ok:
        print(f"  ok       {name:<28} {dt:6.1f}s  {n} figure(s)")
    for name, why in skipped:
        print(f"  skipped  {name:<28} {why}")
    for name in failed:
        print(f"  FAILED   {name}")
    print(f"\n{len(ok)} ran, {len(skipped)} skipped, {len(failed)} failed; "
          f"{len(all_paths)} figures written.")
    if args.smoke and (ok or skipped):
        print("\nNOTE: --smoke figures use tiny sample sizes. They are for "
              "checking the pipeline, never for the report or a progress log.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
