#!/usr/bin/env python3
"""Coordinate reproducible research records; performs no GPU or network work.

Execution is provided separately by research_pipeline.py. This CLI never issues
evaluation receipts, changes a compiler, kills a job, or promotes from booleans.
"""
import argparse
import json
from pathlib import Path
import sys

try:
    from .research_store import Invalid, Store, read_json
except ImportError:
    from research_store import Invalid, Store, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Durable record root; required here or via EMMY_RESEARCH_ROOT")
    parser.add_argument("--work-root", type=Path, help="Mutable workspace root")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create evidence store from baseline source tree")
    init.add_argument("--baseline", required=True, type=Path)
    commands.add_parser("status")
    search = commands.add_parser("search")
    search.add_argument("query")
    propose = commands.add_parser("propose", help="Search prior work and record a research brief")
    propose.add_argument("brief", type=Path)
    mechanism = commands.add_parser("mechanism", help="Append evidence-backed mechanism findings")
    mechanism.add_argument("record", type=Path)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("experiment")
    freeze.add_argument("source", type=Path)
    workspace = commands.add_parser("workspace", help="Materialize a plain isolated baseline workspace")
    workspace.add_argument("experiment")
    restore = commands.add_parser("restore", help="Replay a verified source snapshot into a new directory")
    restore.add_argument("object_id")
    restore.add_argument("destination", type=Path)
    suite = commands.add_parser("suite", help="Register an immutable evaluation contract")
    suite.add_argument("version")
    suite.add_argument("contract", type=Path)
    confirmation = commands.add_parser("reserve-confirmation")
    confirmation.add_argument("experiment")
    confirmation.add_argument("suite_id")
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("experiment")
    enqueue.add_argument("--hypothesis")
    resume = commands.add_parser("resume")
    resume.add_argument("experiment", nargs="?")
    handoff = commands.add_parser("handoff")
    handoff.add_argument("experiment")
    handoff.add_argument("record", type=Path)
    start = commands.add_parser("start-run")
    start.add_argument("run_id")
    start.add_argument("--mode", choices=("smoke", "qualification", "research"), default="smoke")
    activate=commands.add_parser('activate-discovery',help='Record explicit user opt-in; never needed for routine repair')
    activate.add_argument('--reason',required=True)
    deactivate=commands.add_parser('deactivate-discovery',help='Stop admitting research work; never kills unknown processes')
    deactivate.add_argument('--reason',default='explicit stop')
    finish=commands.add_parser('finish-run',help='Close a run without fabricating completion or killing jobs')
    finish.add_argument('run_id',nargs='?'); finish.add_argument('--reason',default='run completed')
    decide = commands.add_parser("decide")
    decide.add_argument("experiment")
    decide.add_argument("decision", choices=("shortlist", "promote", "reject", "inconclusive"))
    decide.add_argument("--reason", required=True)
    decide.add_argument("--evidence-id")
    decide.add_argument("--performance-debt")
    coverage = commands.add_parser("coverage")
    coverage.add_argument("item")
    coverage.add_argument("--experiment")
    coverage.add_argument("--evidence-id")
    coverage.add_argument("--reopen-reason")
    export = commands.add_parser("export")
    export.add_argument("experiment")
    export.add_argument("--inbox", type=Path, default=Path("candidates"))
    verify = commands.add_parser("verify")
    verify.add_argument("object_id")
    gate=commands.add_parser('published-gate',help='Verify an existing exact-source/suite all15 correctness gate')
    gate.add_argument('source_id'); gate.add_argument('suite_id')
    args = parser.parse_args(argv)
    try:
        store = Store(args.root, args.work_root)
        if args.command == "init":
            result = store.init(args.baseline)
        elif args.command == "status":
            result = store.status()
        elif args.command == "search":
            result = store.search(args.query)
        elif args.command == "propose":
            result = store.propose(read_json(args.brief))
        elif args.command == "mechanism":
            result = store.record_mechanism(read_json(args.record))
        elif args.command == "freeze":
            result = store.freeze(args.experiment, args.source)
        elif args.command == "workspace":
            result = store.workspace(args.experiment)
        elif args.command == "restore":
            result = store.restore(args.object_id, args.destination)
        elif args.command == "suite":
            result = store.register_suite(args.version, read_json(args.contract),
                                          {args.contract.name: args.contract.read_bytes()})
        elif args.command == "reserve-confirmation":
            result = store.reserve_confirmation(args.experiment, args.suite_id)
        elif args.command == "enqueue":
            result = store.enqueue(args.experiment, args.hypothesis)
        elif args.command == "resume":
            result = store.resume(args.experiment)
        elif args.command == "handoff":
            result = store.handoff(args.experiment, read_json(args.record))
        elif args.command == "start-run":
            result = store.start_run(args.run_id, args.mode)
        elif args.command == 'activate-discovery':
            result = store.activate_discovery(args.reason)
        elif args.command == 'deactivate-discovery':
            result = store.deactivate_discovery(args.reason)
        elif args.command == 'finish-run':
            result = store.finish_run(args.run_id,args.reason)
        elif args.command == "decide":
            result = store.decide(args.experiment, args.decision, args.reason,
                                  args.evidence_id, args.performance_debt)
        elif args.command == "coverage":
            result = store.coverage(args.item, args.experiment, args.evidence_id, args.reopen_reason)
        elif args.command == "export":
            result = store.export(args.experiment, args.inbox)
        elif args.command == "verify":
            result = store.verify_object(args.object_id)
        elif args.command == 'published-gate':
            result = store.require_published_gate(args.source_id,args.suite_id)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (Invalid, OSError, ValueError, KeyError) as error:
        print(f"research: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
