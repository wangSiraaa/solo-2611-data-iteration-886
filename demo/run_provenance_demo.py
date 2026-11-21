#!/usr/bin/env python3
"""End-to-end demo: same-origin revisions, exposure ledger, eval packages.

Run from the repository root:

    python3 demo/run_provenance_demo.py

It builds a tiny synthetic source (no model, no training), replays the worked
scenarios from the task -- X-M-B evidence through a retracted-body node M,
parallel edges with revoke-one semantics, signed-package recheck/reissue, and
the loan-window quarantine timeline (add a / add b / revoke a / revoke b) --
and prints every decision with the evidence witness behind it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from wrsource.evidence import EvidenceLedger
from wrsource.loan import loan_timeline, loaned_package_from_manifest
from wrsource.package import (PackageRegistry, recheck_package,
                              reissue_clean_package, selected_row_ids)
from wrsource.source import SourceService
from wrsource.training import TrainingLedger

SEED = bytes(32)


def banner(text):
    print("\n" + "=" * 74)
    print(text)
    print("=" * 74)


def show(title, obj):
    print(f"\n-- {title}")
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def add_row(svc, row_id, group, body):
    svc.apply_batch([{"op": "add", "row_id": row_id, "group": group,
                      "weight": 50, "body": body, "shard": "s0"}])


def main():
    work = os.path.join(HERE, "_work_provenance")
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)

    # ---- 1. Synthetic source ------------------------------------------------
    banner("1. 合成源：g1 = A, B, M（均权），g2 = X；零种子下 B 排在 A 前")
    svc = SourceService(os.path.join(work, "source.sqlite"))
    svc.init()
    for rid, grp, body in [("A", "g1", "text alpha"),
                           ("B", "g1", "text beta"),
                           ("M", "g1", "text mmm"),
                           ("X", "g2", "text xray"),
                           ("C", "g1", "text gamma")]:
        add_row(svc, rid, grp, body)
    version = svc.latest_version()
    print("source version:", version, "| groups:",
          svc.group_counts(version))

    evidence = EvidenceLedger(os.path.join(work, "evidence.sqlite"))
    evidence.init()
    training = TrainingLedger(os.path.join(work, "training.sqlite"))
    training.init()
    training.create_plan("T")
    training.create_plan("T-empty")
    training.register_batch("T", ["X"], source_version=version,
                            evidence_revision=0,
                            note="训练产品实际发出的原始行身份 X")
    registry = PackageRegistry(os.path.join(work, "packages.sqlite"))
    registry.init()
    quotas = {"g1": 1}

    # ---- 2. X-M-B with M retracted ------------------------------------------
    banner("2. 证据 X–M–B；M 正文撤回后节点仍连通，T 的评测选 A")
    r1 = evidence.add_evidence("ev-xm", "X", "M", note="X 与 M 同原件")
    r2 = evidence.add_evidence("ev-mb", "M", "B", note="M 与 B 同原件")
    print("evidence revisions:", r1, r2,
          "| 同源路径 X -> B:",
          evidence.origin_path("X", "B").as_dict())
    svc.apply_batch([{"op": "retract", "row_id": "M"}])
    v2 = svc.latest_version()
    print("M 撤回后源版本:", v2, "| 证据图仍判 X 与 B 同源:",
          evidence.same_origin("X", "B"))
    pkg_t = registry.issue(svc, v2, SEED, quotas, training, "T", evidence,
                           evidence.latest_revision(),
                           note="计划 T，固定三元组签发")
    print("T 选中:", selected_row_ids(pkg_t))
    show("被剔除候选及其证据链",
         pkg_t["groups"]["g1"]["excluded"])

    banner("3. 证据不动，换空训练计划 T-empty：没有暴露，选 B")
    pkg_empty = registry.issue(svc, v2, SEED, quotas, training, "T-empty",
                               evidence, evidence.latest_revision())
    print("T-empty 选中:", selected_row_ids(pkg_empty))

    banner("4. 固定 T，撤销唯一的 M–B 证据后重新签发：选 B；A/B 仍是两条记录")
    r3 = evidence.revoke_evidence("ev-mb", note="来源修正：撤回 M-B")
    print("撤销后修订号:", r3, "| X 与 B 仍同源:",
          evidence.same_origin("X", "B"))
    pkg_t2 = registry.issue(svc, v2, SEED, quotas, training, "T", evidence,
                            evidence.latest_revision())
    print("T 重新签发选中:", selected_row_ids(pkg_t2))
    print("A 与 B 同正文也不会自动同源:", evidence.same_origin("A", "B"))
    print("旧版重开（rev 2）X-B 同源仍可查:", evidence.same_origin("X", "B", 2))

    # ---- 5. Signed package recheck and clean reissue ------------------------
    banner("5. 已签包反查：补录证据使旧成员 B 需要隔离，旧包判断保留")
    signed = registry.issue(svc, version, SEED, quotas, training, "T",
                            evidence, 0, note="rev0 时 B 干净，已签")
    print("已签包", signed["package_id"], "选中:",
          selected_row_ids(signed))
    evidence.add_evidence("ev-xb", "X", "B", note="补录：X 与 B 同原件")
    report = recheck_package(registry.open(signed["package_id"]),
                             training, evidence)
    show("复查结果（旧包不可变）", {
        "package_id": report["package_id"],
        "signed_rev": report["signed_evidence_revision"],
        "checked_rev": report["checked_evidence_revision"],
        "quarantine": report["quarantine_required"],
        "still_clean": report["still_clean"]})
    new_pkg, reissue = reissue_clean_package(
        registry, svc, signed["package_id"], training, evidence,
        note="补录后重新抽足的干净包")
    print("新干净包", new_pkg["package_id"], "选中:",
          selected_row_ids(new_pkg), "| 父包:",
          new_pkg["parent_package_id"])
    show("重发补位账", {
        "dropped": reissue["dropped_rows"],
        "backfilled": reissue["backfilled_rows"]})
    print("封存包数量:", len(registry.list_packages()),
          "| 旧包仍记录:",
          selected_row_ids(registry.open(signed["package_id"])))

    # ---- 6. Parallel edges ---------------------------------------------------
    banner("6. 重复边：同一对端点两份独立证据，撤一份不断开")
    evidence.add_evidence("dup-a", "X", "B")
    evidence.add_evidence("dup-b", "B", "X")
    print("两份证据时同源:", evidence.same_origin("X", "B"),
          "| 该对证据:",
          [e.evidence_id for e in evidence.evidence_between("X", "B")])
    evidence.revoke_evidence("dup-a")
    print("撤 dup-a 后仍同源:", evidence.same_origin("X", "B"))
    evidence.revoke_evidence("dup-b")
    print("撤 dup-b 后断开:", not evidence.same_origin("X", "B"))

    # ---- 7. Split then reconnect --------------------------------------------
    banner("7. 大组先分裂再被另一证据连回")
    evidence.add_evidence("link-xm2", "X", "M")
    evidence.add_evidence("link-mb2", "M", "B")
    print("连通:", evidence.same_origin("X", "B"))
    evidence.revoke_evidence("link-mb2")
    print("撤 M-B 后分裂:", not evidence.same_origin("X", "B"))
    evidence.add_evidence("link-xb2", "X", "B")
    print("新证据 X-B 连回:", evidence.same_origin("X", "B"),
          "| 路径:",
          evidence.origin_path("M", "B").as_dict())

    # ---- 8. Loan-window quarantine timeline ---------------------------------
    banner("8. 借阅窗口：起点安全；add a 隔离、add b 无重复动作、撤 a 不动、撤 b 解封")
    # Fresh ledger so the revision sequence is exactly 1..4 like the task.
    ev2 = EvidenceLedger(os.path.join(work, "evidence_loan.sqlite"))
    ev2.init()
    loan_pkg = registry.issue(svc, version, SEED, {"g1": 2}, training,
                              "T-empty", ev2, 0, note="外借包")
    loan = loaned_package_from_manifest("loan-B-A", loan_pkg)
    ev2.add_evidence("a", "X", "B", note="新建证据 a")
    ev2.add_evidence("b", "X", "B", note="同端点独立证据 b")
    ev2.revoke_evidence("a", note="撤 a，b 仍支持")
    ev2.revoke_evidence("b", note="撤 b，最后一条断开")
    timeline = loan_timeline(loan, "T", 0, ev2.latest_revision(),
                             training, ev2, cutoff_batch=1)
    show("计划 T 的隔离时间线", timeline.as_dict())

    timeline_other = loan_timeline(loan, "T-empty", 0,
                                   ev2.latest_revision(), training, ev2,
                                   cutoff_batch=0)
    print("\n无 X 暴露的计划 T-empty 动作数:",
          len(timeline_other.events), "(应为 0)")
    print("\n工作目录:", work)
    print("可用 CLI 入口: python -m wrsource.evidence / wrsource.training / "
          "wrsource.package / wrsource.loan --help")


if __name__ == "__main__":
    main()
