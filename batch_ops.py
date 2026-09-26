#!/usr/bin/env python3
"""嵌套批量操作执行器（纯 Python 标准库，单文件）。

用法：
    python3 batch_ops.py              # 运行内置自测样例
    python3 batch_ops.py spec.json    # 执行 JSON 文件描述的操作流并输出报告

====================== 设计说明（自定规则及理由） ======================

【数据模型】
- 目标状态是一个 dict：target(字符串) -> value(数值/字符串等 JSON 值)。
- 操作是一棵树：每个操作可含 children 列表。父子层级语义为
  “父操作建立资源/上下文，子操作在其上细化”（例如父 set 创建账户，
  子 increment 调整余额）。采用前序应用：先应用父操作自身，再按序应用
  子操作。理由：子操作通常依赖父操作产生的前提（目标已存在），前序最
  符合直觉且让前置校验（如 increment 要求目标存在）自然生效。

【操作类型与前置校验】
- set:        写入 target=value。前置：必须带 value 字段。
- delete:     删除 target。前置：target 必须已存在，否则报 TARGET_NOT_FOUND。
- increment:  target += amount（amount 缺省为 1）。前置：target 已存在且
              当前值为数值，否则报 TARGET_NOT_FOUND / PRECONDITION_FAILED。

【失败与级联回滚范围】
- 任一操作失败（前置校验失败 / 目标不存在 / 冲突 / 超深 / 未知类型），
  该操作记错误（含操作路径），并视为其所在父操作失败。
- 回滚级联范围 = 失败节点的整个祖先链直至根，且包含各层已应用的兄弟
  子树，即整批原子回滚。理由：父操作的语义完整性依赖全部子操作成功
  （“父失败子操作全部回滚、子失败父操作已应用部分回滚”），逐层向上
  传播后等价于整批 all-or-nothing；部分提交会留下语义不一致的中间态。
- 回滚按应用逆序执行，每个操作在应用时记录逆操作（undo），保证回滚后
  状态与原状态完全一致、无残留。

【冲突规则】
- 同一批次内，两个写操作（set/delete/increment）作用于同一 target 即
  为冲突：后到的操作报 CONFLICT 并失败（触发级联回滚）。理由：同批次
  内对同一目标的多个写结果依赖顺序，属于规格歧义，拒绝比静默覆盖安全。

【嵌套深度】
- 根操作深度为 1，超过 MAX_DEPTH（默认 8）报 DEPTH_EXCEEDED 并失败。

【输出】
- 应用日志（按应用顺序，含操作路径）、回滚日志（逆序）、最终状态、
  错误清单（路径 + 类型 + 说明）。
======================================================================
"""

from __future__ import annotations

import copy
import json
import sys

MAX_DEPTH = 8  # 嵌套深度上限（根操作深度为 1）

OP_TYPES = ("set", "delete", "increment")


class Engine:
    """对一份状态应用嵌套操作流，失败时级联回滚。"""

    def __init__(self, state, max_depth=MAX_DEPTH):
        self.state = state
        self.max_depth = max_depth
        self.errors = []        # [{path, type, message}]
        self.apply_log = []     # 应用日志（按应用顺序）
        self.rollback_log = []  # 回滚日志（按回滚顺序，即应用逆序）
        self.touched = {}       # target -> 本批次第一个写它的操作路径（冲突检测）

    # ---------------- 对外入口 ----------------

    def run(self, operations):
        """应用整批操作；任一失败则级联回滚至原状态。返回报告 dict。"""
        journal = []  # 已成功根操作的 undo 条目（用于整批回滚）
        for index, op in enumerate(operations):
            path = "/" + str(op.get("id", f"op{index}"))
            node_journal = []
            if not self._apply_node(op, path, 1, node_journal):
                # 级联：先回滚当前根子树已应用部分，再回滚之前已成功的根
                self._rollback(node_journal)
                self._rollback(journal)
                return self._report(success=False)
            journal.extend(node_journal)
        return self._report(success=True)

    # ---------------- 内部实现 ----------------

    def _apply_node(self, op, path, depth, journal):
        """应用单个操作节点（先自身，后按序子操作）。失败返回 False。

        journal 为本节点（含其子树）的 undo 条目列表；失败时由调用方
        负责回滚该 journal（实现逐层级联）。
        """
        if depth > self.max_depth:
            self._error(path, "DEPTH_EXCEEDED",
                        f"嵌套深度 {depth} 超过上限 {self.max_depth}")
            return False

        undo_entry = self._apply_action(op, path)
        if undo_entry is None:
            return False  # 自身前置校验/冲突失败，错误已记录
        journal.append(undo_entry)

        for index, child in enumerate(op.get("children", [])):
            child_path = path + "/" + str(child.get("id", f"op{index}"))
            child_journal = []
            if not self._apply_node(child, child_path, depth + 1, child_journal):
                # 子操作失败：立即回滚子树已应用部分，本节点视为失败，
                # 本节点自身及已应用兄弟由调用方回滚（级联向上）。
                self._rollback(child_journal)
                return False
            journal.extend(child_journal)
        return True

    def _apply_action(self, op, path):
        """校验并应用单个动作，成功返回 undo 条目，失败记录错误返回 None。"""
        otype = op.get("type")
        target = op.get("target")

        if otype not in OP_TYPES:
            self._error(path, "UNKNOWN_TYPE", f"未知操作类型: {otype!r}")
            return None
        if not isinstance(target, str) or not target:
            self._error(path, "INVALID_TARGET", "操作缺少有效的 target 字段")
            return None

        # 冲突检测：同批次内两个写操作作用于同一目标
        if target in self.touched:
            self._error(path, "CONFLICT",
                        f"目标 {target!r} 与本批次先前的操作 "
                        f"{self.touched[target]} 冲突")
            return None

        if otype == "set":
            if "value" not in op:
                self._error(path, "PRECONDITION_FAILED",
                            "set 操作缺少 value 字段")
                return None
            existed = target in self.state
            old_value = self.state.get(target)
            self.state[target] = op["value"]
            if existed:
                undo = lambda t=target, v=old_value: self.state.__setitem__(t, v)
                desc = f"恢复 {target!r} 为旧值 {old_value!r}"
            else:
                undo = lambda t=target: self.state.pop(t, None)
                desc = f"删除新建的 {target!r}"
            detail = f"set {target!r} = {op['value']!r}"

        elif otype == "delete":
            if target not in self.state:
                self._error(path, "TARGET_NOT_FOUND",
                            f"delete 的目标 {target!r} 不存在")
                return None
            old_value = self.state.pop(target)
            undo = lambda t=target, v=old_value: self.state.__setitem__(t, v)
            desc = f"恢复被删除的 {target!r} = {old_value!r}"
            detail = f"delete {target!r}（原值 {old_value!r}）"

        else:  # increment
            if target not in self.state:
                self._error(path, "TARGET_NOT_FOUND",
                            f"increment 的目标 {target!r} 不存在")
                return None
            current = self.state[target]
            if isinstance(current, bool) or not isinstance(current, (int, float)):
                self._error(path, "PRECONDITION_FAILED",
                            f"increment 的目标 {target!r} 当前值 {current!r} 不是数值")
                return None
            amount = op.get("amount", 1)
            if isinstance(amount, bool) or not isinstance(amount, (int, float)):
                self._error(path, "PRECONDITION_FAILED",
                            f"increment 的 amount {amount!r} 不是数值")
                return None
            self.state[target] = current + amount
            undo = lambda t=target, a=amount: self.state.__setitem__(t, self.state[t] - a)
            desc = f"将 {target!r} 回减 {amount!r}"
            detail = f"increment {target!r} += {amount!r}（{current!r} -> {self.state[target]!r}）"

        self.touched[target] = path
        self.apply_log.append(f"{path}: {detail}")
        return (path, desc, undo)

    def _rollback(self, journal):
        """按应用逆序回滚 journal 中的 undo 条目。"""
        for path, desc, undo in reversed(journal):
            undo()
            self.rollback_log.append(f"{path}: {desc}")

    def _error(self, path, kind, message):
        self.errors.append({"path": path, "type": kind, "message": message})

    def _report(self, success):
        return {
            "success": success,
            "applied": list(self.apply_log),
            "rolled_back": list(self.rollback_log),
            "final_state": copy.deepcopy(self.state),
            "errors": copy.deepcopy(self.errors),
        }


def execute(spec, max_depth=MAX_DEPTH):
    """执行 {'state': {...}, 'operations': [...]} 规格的便捷函数。"""
    engine = Engine(copy.deepcopy(spec.get("state", {})), max_depth=max_depth)
    return engine.run(spec.get("operations", []))


# ============================ 自测样例 ============================

def _deep_chain(depth):
    """生成 depth 层嵌套的 set 操作链。"""
    node = {"id": f"n{depth}", "type": "set", "target": f"t{depth}", "value": depth}
    for level in range(depth - 1, 0, -1):
        node = {"id": f"n{level}", "type": "set", "target": f"t{level}",
                "value": level, "children": [node]}
    return node


def run_self_tests():
    cases = []

    # 样例 1：三层嵌套全部成功（父调余额，子增计数，孙记标记）
    cases.append(("样例1-全部成功", {
        "state": {"acct": 100, "counter": 0},
        "operations": [
            {"id": "deposit", "type": "increment", "target": "acct", "amount": 50,
             "children": [
                 {"id": "bump_counter", "type": "increment", "target": "counter",
                  "children": [
                      {"id": "mark", "type": "set", "target": "acct.audited",
                       "value": True},
                  ]},
                 {"id": "tag", "type": "set", "target": "acct.tag", "value": "vip"},
             ]},
        ],
    }, lambda r, spec: (
        r["success"]
        and r["final_state"] == {"acct": 150, "counter": 1,
                                 "acct.audited": True, "acct.tag": "vip"}
        and not r["errors"] and not r["rolled_back"]
        and len(r["applied"]) == 4
    )))

    # 样例 2：深层子操作引用不存在的目标 -> 级联回滚整批，状态无残留
    spec2 = {
        "state": {"base": 1},
        "operations": [
            {"id": "root_a", "type": "set", "target": "a", "value": 1,
             "children": [
                 {"id": "child_a1", "type": "set", "target": "a1", "value": 2},
                 {"id": "child_a2", "type": "increment", "target": "ghost"},
             ]},
            {"id": "root_b", "type": "set", "target": "b", "value": 3},
        ],
    }
    cases.append(("样例2-子操作失败级联回滚", spec2, lambda r, spec: (
        not r["success"]
        and r["final_state"] == spec["state"]            # 回滚后与原状态一致
        and any(e["type"] == "TARGET_NOT_FOUND"
                and e["path"] == "/root_a/child_a2" for e in r["errors"])
        and len(r["rolled_back"]) == 2                   # root_a 与 child_a1 被回滚
        and "b" not in r["final_state"]                  # root_b 从未应用
    )))

    # 样例 3：同目标冲突（父 set 后子 delete 同一目标）
    spec3 = {
        "state": {},
        "operations": [
            {"id": "p", "type": "set", "target": "x", "value": 1,
             "children": [
                 {"id": "c", "type": "delete", "target": "x"},
             ]},
        ],
    }
    cases.append(("样例3-同目标冲突", spec3, lambda r, spec: (
        not r["success"]
        and any(e["type"] == "CONFLICT" and e["path"] == "/p/c"
                for e in r["errors"])
        and r["final_state"] == spec["state"]
    )))

    # 样例 4：嵌套深度超过上限
    spec4 = {"state": {}, "operations": [_deep_chain(MAX_DEPTH + 2)]}
    cases.append(("样例4-深度超限", spec4, lambda r, spec: (
        not r["success"]
        and any(e["type"] == "DEPTH_EXCEEDED" for e in r["errors"])
        and r["final_state"] == spec["state"]
        and len(r["rolled_back"]) == MAX_DEPTH           # 超限前已应用的全部回滚
    )))

    # 样例 5：delete 不存在的目标（前置校验失败），前面根操作也被级联回滚
    spec5 = {
        "state": {"keep": 7},
        "operations": [
            {"id": "first", "type": "set", "target": "tmp", "value": 1},
            {"id": "second", "type": "delete", "target": "missing"},
        ],
    }
    cases.append(("样例5-目标不存在且整批回滚", spec5, lambda r, spec: (
        not r["success"]
        and any(e["type"] == "TARGET_NOT_FOUND" and e["path"] == "/second"
                for e in r["errors"])
        and r["final_state"] == spec["state"]            # first 的写入也被回滚
        and r["rolled_back"]                             # 有回滚记录
    )))

    # 样例 6：increment 非数值目标（前置校验）
    spec6 = {
        "state": {"name": "abc"},
        "operations": [{"id": "bad", "type": "increment", "target": "name"}],
    }
    cases.append(("样例6-前置校验失败", spec6, lambda r, spec: (
        not r["success"]
        and any(e["type"] == "PRECONDITION_FAILED" for e in r["errors"])
        and r["final_state"] == spec["state"]
    )))

    all_passed = True
    for name, spec, check in cases:
        report = execute(spec)
        passed = check(report, spec)
        all_passed = all_passed and passed
        print("=" * 66)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print("=" * 66)
    print("自测结果:", "全部通过" if all_passed else "存在失败用例")
    return all_passed


def main(argv):
    if len(argv) > 1:
        with open(argv[1], encoding="utf-8") as fh:
            spec = json.load(fh)
        report = execute(spec)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["success"] else 1
    return 0 if run_self_tests() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
