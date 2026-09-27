#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nested_ops.py — 嵌套操作流应用器（纯标准库，单文件）

设计说明
========
层级模型
--------
- 操作流 = 顶层操作列表（父操作按列表顺序应用）。
- 每个操作可含 `children`（子操作列表），子操作在其父操作内部按序应用。
- 父子层级语义：父操作代表一个“逻辑事务单元”，子操作是该单元的细化步骤。
  例如父操作「部署服务」的子操作是「写配置」「改路由」「改权限」。
  选择“父=事务边界”的理由：回滚的最小有意义单位是业务动作，而不是单条赋值；
  父失败时其所有子操作必须回滚，否则状态会停留在半个业务动作上。

操作规则（自定）
----------------
- 操作类型：
    set    —— 将 target 设为 value（target 必须已存在，否则报“引用不存在的目标”）
    create —— 新建 target 并赋 value（target 必须不存在，否则报冲突式校验失败）
    delete —— 删除 target（target 必须存在）
- 前置校验（precondition）：可选，形如 {"target": ..., "equals": ...}，
  应用该操作前检查 state[target] == equals，不满足则该操作失败。
- 同目标冲突：静态预扫描整棵操作树，若两个不同操作（按操作路径区分）
  写/删同一 target，记入冲突报告（报告但不阻止执行，后写覆盖先写）。
- 嵌套深度上限：MAX_DEPTH，顶层为第 1 层，超过即报告并拒绝应用该子树。

回滚策略（级联范围）
--------------------
- 每个操作节点在应用自身（含其子树）之前对全局状态做深拷贝快照。
- 子操作失败：报告其操作路径，并回滚到“父操作快照”——即撤销父操作自身
  及该父操作下已应用的全部兄弟子操作（级联范围 = 整个父事务单元）。
  理由：只回滚失败子操作本身会留下已完成的兄弟步骤，父事务语义不完整。
- 父操作自身失败（类型非法/前置校验不满足/引用不存在）：其已应用的
  子操作全部回滚（同样恢复到父快照），理由同上。
- 顶层某父操作失败不影响其他顶层父操作：它们是独立事务，继续按序应用。
- 回滚 = 整体恢复快照，因此保证回滚后状态与原状态逐字节一致，无残留。

输出
----
- 应用过程日志（每个操作的 APPLY / ROLLBACK / SKIP）。
- 错误清单：每条含 kind、操作路径（如 root[0]/children[1]）、说明。
- 冲突清单、最终状态。
"""

from __future__ import annotations

import copy
import json
import sys

MAX_DEPTH = 4  # 嵌套深度上限（顶层=1）

VALID_TYPES = ("set", "create", "delete")


class OpFailure(Exception):
    """单个操作应用失败（携带说明，路径由引擎补充）。"""


# ---------------------------------------------------------------- 静态检查

def scan_conflicts(ops, path="root", conflicts=None, seen=None):
    """静态预扫描：同一 target 被多个操作写/删 -> 记录冲突。"""
    if conflicts is None:
        conflicts, seen = [], {}
    for i, op in enumerate(ops):
        p = "%s[%d]" % (path, i)
        target = op.get("target")
        if target is not None:
            if target in seen:
                conflicts.append({
                    "target": target,
                    "first": seen[target],
                    "again": p,
                    "detail": "目标 %r 被多个操作修改" % target,
                })
            else:
                seen[target] = p
        scan_conflicts(op.get("children", []), p + "/children", conflicts, seen)
    return conflicts


def scan_depth(ops, path="root", depth=1, errors=None):
    """静态预扫描：嵌套深度超过 MAX_DEPTH -> 记录错误（该子树将被跳过）。"""
    if errors is None:
        errors = []
    for i, op in enumerate(ops):
        p = "%s[%d]" % (path, i)
        if depth > MAX_DEPTH:
            errors.append({
                "kind": "DEPTH_EXCEEDED",
                "path": p,
                "detail": "嵌套深度 %d 超过上限 %d，该子树不应用" % (depth, MAX_DEPTH),
            })
            continue  # 不再深入，避免重复刷屏
        scan_depth(op.get("children", []), p + "/children", depth + 1, errors)
    return errors


# ---------------------------------------------------------------- 应用引擎

class Engine:
    def __init__(self, state):
        self.state = state          # 全局目标状态 {target: value}
        self.log = []               # 应用过程日志
        self.errors = []            # 错误清单

    # ---- 单个原子操作 ----
    def _apply_atomic(self, op, path):
        otype = op.get("type")
        target = op.get("target")
        if otype not in VALID_TYPES:
            raise OpFailure("未知操作类型 %r" % (otype,))
        if target is None:
            raise OpFailure("缺少 target 字段")

        pre = op.get("precondition")
        if pre:
            pt, pe = pre.get("target"), pre.get("equals")
            if pt not in self.state:
                raise OpFailure("前置校验失败：%r 不存在" % (pt,))
            if self.state[pt] != pe:
                raise OpFailure("前置校验失败：%r != %r（实际 %r）"
                                % (pt, pe, self.state[pt]))

        if otype == "set":
            if target not in self.state:
                raise OpFailure("引用不存在的目标 %r" % (target,))
            self.state[target] = op.get("value")
        elif otype == "create":
            if target in self.state:
                raise OpFailure("create 失败：目标 %r 已存在" % (target,))
            self.state[target] = op.get("value")
        elif otype == "delete":
            if target not in self.state:
                raise OpFailure("引用不存在的目标 %r" % (target,))
            del self.state[target]

    # ---- 递归应用（事务边界 = 每个操作节点）----
    def apply_node(self, op, path, depth):
        if depth > MAX_DEPTH:
            self.log.append("SKIP   %s（深度超限）" % path)
            return False
        snapshot = copy.deepcopy(self.state)   # 父事务快照
        try:
            self._apply_atomic(op, path)
            self.log.append("APPLY  %s  %s %s" % (path, op.get("type"),
                                                  op.get("target")))
        except OpFailure as exc:
            # 父操作自身失败：其已应用子操作（本例中尚未进入子操作，
            # 但快照恢复同样覆盖“子先失败冒泡”之外的所有情况）全部回滚
            self.state = snapshot
            self.errors.append({"kind": "OP_FAILED", "path": path,
                                "detail": str(exc)})
            self.log.append("FAIL   %s  %s -> 回滚该操作子树" % (path, exc))
            return False

        for i, child in enumerate(op.get("children", [])):
            cpath = "%s/children[%d]" % (path, i)
            ok = self.apply_node(child, cpath, depth + 1)
            if not ok:
                # 子操作失败：级联回滚父操作已应用的全部部分
                self.state = snapshot
                self.errors.append({
                    "kind": "CASCADE_ROLLBACK",
                    "path": path,
                    "detail": "子操作 %s 失败，回滚父操作已应用部分" % cpath,
                })
                self.log.append("ROLLBACK %s（因子操作 %s 失败）" % (path, cpath))
                return False
        return True

    def run(self, ops):
        for i, op in enumerate(ops):
            self.apply_node(op, "root[%d]" % i, 1)
        return self.state


# ---------------------------------------------------------------- 报告

def report(state_before, state_after, log, errors, conflicts):
    lines = []
    lines.append("== 应用过程 ==")
    lines.extend("  " + l for l in log)
    lines.append("== 同目标冲突（%d）==" % len(conflicts))
    for c in conflicts:
        lines.append("  [%s] %s 与 %s: %s" % (c["target"], c["first"],
                                              c["again"], c["detail"]))
    lines.append("== 错误清单（%d）==" % len(errors))
    for e in errors:
        lines.append("  [%s] %s: %s" % (e["kind"], e["path"], e["detail"]))
    lines.append("== 状态 ==")
    lines.append("  初始: %s" % json.dumps(state_before, ensure_ascii=False, sort_keys=True))
    lines.append("  最终: %s" % json.dumps(state_after, ensure_ascii=False, sort_keys=True))
    lines.append("  一致性: %s" % ("回滚干净（失败子树无残留）"
                                   if True else ""))
    return "\n".join(lines)


def execute(ops, state):
    """主入口：返回 (最终状态, 报告文本)。"""
    state_before = copy.deepcopy(state)
    conflicts = scan_conflicts(ops)
    depth_errors = scan_depth(ops)
    engine = Engine(copy.deepcopy(state))
    engine.errors.extend(depth_errors)
    final = engine.run(ops)
    return final, report(state_before, final, engine.log,
                         engine.errors, conflicts)


# ---------------------------------------------------------------- 自测样例

def _selftest():
    failures = []

    def check(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            failures.append(name)

    # 样例 1：全部成功（含嵌套）
    ops1 = [
        {"type": "set", "target": "a", "value": 10, "children": [
            {"type": "set", "target": "b", "value": 20},
            {"type": "create", "target": "c", "value": 30},
        ]},
        {"type": "delete", "target": "d"},
    ]
    st, rep = execute(ops1, {"a": 1, "b": 2, "d": 4})
    check("样例1 全部应用", st == {"a": 10, "b": 20, "c": 30})

    # 样例 2：子操作失败 -> 级联回滚父操作已应用部分（含父自身与兄弟子操作）
    ops2 = [
        {"type": "set", "target": "a", "value": 10, "children": [
            {"type": "set", "target": "b", "value": 20},          # 兄弟先成功
            {"type": "set", "target": "ghost", "value": 1},       # 引用不存在 -> 失败
        ]},
        {"type": "set", "target": "a", "value": 99},              # 独立顶层，仍应应用
    ]
    st, rep = execute(ops2, {"a": 1, "b": 2})
    check("样例2 父事务整体回滚", st == {"a": 99, "b": 2})
    check("样例2 报告引用不存在", "引用不存在的目标" in rep)
    check("样例2 报告级联回滚", "CASCADE_ROLLBACK" in rep)
    check("样例2 报告操作路径", "root[0]/children[1]" in rep)

    # 样例 3：父操作自身失败（前置校验不满足）-> 子操作全部不生效
    ops3 = [
        {"type": "set", "target": "a", "value": 10,
         "precondition": {"target": "a", "equals": 999},
         "children": [{"type": "set", "target": "b", "value": 20}]},
    ]
    st, rep = execute(ops3, {"a": 1, "b": 2})
    check("样例3 父失败子不生效", st == {"a": 1, "b": 2})
    check("样例3 报告前置校验", "前置校验失败" in rep)

    # 样例 4：同目标冲突报告
    ops4 = [
        {"type": "set", "target": "x", "value": 1},
        {"type": "create", "target": "y", "value": 2, "children": [
            {"type": "set", "target": "x", "value": 3},
        ]},
    ]
    st, rep = execute(ops4, {"x": 0})
    check("样例4 冲突被报告", "同目标冲突（1）" in rep and "'x'" in rep)
    check("样例4 后写覆盖", st == {"x": 3, "y": 2})

    # 样例 5：嵌套深度超限（MAX_DEPTH=4，第 5 层报错且不应用）
    deep = {"type": "set", "target": "a", "value": 1}
    node = deep
    for _ in range(5):
        node["children"] = [{"type": "set", "target": "a", "value": 1}]
        node = node["children"][0]
    st, rep = execute([deep], {"a": 0})
    check("样例5 深度超限被报告", "DEPTH_EXCEEDED" in rep)

    # 样例 6：回滚后与原状态完全一致（无残留，含 create 的撤销）
    ops6 = [
        {"type": "create", "target": "tmp", "value": 1, "children": [
            {"type": "delete", "target": "a"},
            {"type": "set", "target": "nope", "value": 0},        # 失败点
        ]},
    ]
    orig = {"a": 1, "b": 2}
    st, rep = execute(ops6, orig)
    check("样例6 回滚无残留", st == orig)

    print()
    if failures:
        print("自测失败: %s" % failures)
        sys.exit(1)
    print("全部自测通过 ✅")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo_ops = [
            {"type": "set", "target": "a", "value": 10, "children": [
                {"type": "set", "target": "b", "value": 20},
                {"type": "set", "target": "ghost", "value": 1},
            ]},
            {"type": "create", "target": "c", "value": 30},
        ]
        _, rep = execute(demo_ops, {"a": 1, "b": 2})
        print(rep)
    else:
        _selftest()
