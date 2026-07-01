"""Output parsing for model predictions."""

import json
import re
from typing import Optional


def parse_v5_structured_json(
    output: str,
) -> tuple[list[list[float]], list[Optional[str]]]:
    """Parse v5 structured JSON output.

    Expected format (inside ```json ... ```):
        [{"point_3d": [u, v, d], "label": "1", "gripper": "close"}, ...]

    Returns:
        (waypoints, actions) where waypoints are [[u, v, depth], ...] and
        actions are ["Close Gripper", "Open Gripper", None, ...]
    """
    m = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
    raw = m.group(1) if m else output.strip()

    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return [], []

    waypoints, actions = [], []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pt = entry.get("point_3d", entry.get("point_2d"))
        if pt is None:
            continue
        waypoints.append(
            [float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 0.0]
        )
        grip = entry.get("gripper", "none")
        if grip == "close":
            actions.append("Close Gripper")
        elif grip == "open":
            actions.append("Open Gripper")
        else:
            actions.append(None)
    return waypoints, actions


def parse_v3_ans_tags(
    output: str,
) -> tuple[list[list[float]], list[Optional[str]]]:
    """Parse v3 format: <ans>[[u, v, d], <action>Close Gripper</action>, ...]</ans>"""
    waypoints, actions = [], []

    ans_match = re.search(r"<ans>(.*?)</ans>", output, re.DOTALL)
    if not ans_match:
        ans_match = re.search(r"\[\[.*?\]\]", output, re.DOTALL)
        if not ans_match:
            return [], []
        content = ans_match.group(0)
    else:
        content = ans_match.group(1)

    coord_pat = r"\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]"
    action_pat = r"<action>(.*?)</action>"
    parts = re.split(action_pat, content)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            for c in re.findall(coord_pat, part):
                waypoints.append([float(c[0]), float(c[1]), float(c[2])])
                actions.append(None)
        else:
            if actions:
                actions[-1] = part.strip()

    return waypoints, actions


def parse_trajectory(
    output: str, prompt_style: str = "v5"
) -> tuple[list[list[float]], list[Optional[str]]]:
    """Auto-detect and parse trajectory from model output.

    Args:
        output: Raw model output string
        prompt_style: "v5" for structured JSON, "v3" for <ans> tags

    Returns:
        (waypoints, actions)
    """
    if prompt_style == "v5":
        wp, act = parse_v5_structured_json(output)
        if wp:
            return wp, act
        # Fall back to v3
        return parse_v3_ans_tags(output)
    else:
        wp, act = parse_v3_ans_tags(output)
        if wp:
            return wp, act
        return parse_v5_structured_json(output)
