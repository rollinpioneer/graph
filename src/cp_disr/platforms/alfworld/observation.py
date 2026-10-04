"""Public-text parsing for ALFWorld/TextWorld feedback. No hidden facts are read here."""
import re

_ITEM = re.compile(r"([a-z]+ \d+)")
_TASK = re.compile(r"put (?:a|an|some) (\w+) (?:in|on)(?:/on)? (\w+)\.")


def type_of(name):
    """'alarmclock 3' -> 'alarmclock' (instance name to class)."""
    return name.rsplit(" ", 1)[0]


def parse_initial(feedback):
    """Initial text: every receptacle instance in the room and the goal classes."""
    flat = feedback.replace("\n", " ")
    seen = re.search(r"you see (.*?)\.\s", flat + " ")
    if seen is None:
        raise ValueError("initial feedback lists no receptacles")
    receptacles = sorted(set(_ITEM.findall(seen.group(1))))
    task = _TASK.search(flat)
    if task is None:
        raise ValueError("initial feedback has no pick-and-place goal")
    return {"receptacles": receptacles, "goal_otype": task.group(1), "goal_rtype": task.group(2)}


def _items_after(flat, marker):
    i = flat.find(marker)
    if i < 0:
        return None
    tail = flat[i + len(marker):]
    if tail.startswith("nothing"):
        return []
    return list(_ITEM.findall(tail.split(".")[0]))


def parse_arrival(feedback):
    """After 'go to r': returns {'at', 'closed', 'items'}; items is None while closed."""
    flat = feedback.replace("\n", " ")
    m = re.search(r"You arrive at ([a-z]+ \d+)\.", flat)
    if m is None:
        return None
    closed = bool(re.search(r"The [a-z]+ \d+ is closed", flat))
    items = None if closed else _items_after(flat, "you see ")
    return {"at": m.group(1), "closed": closed, "items": items}


def parse_examine(feedback, r):
    """After 'examine r' (receptacle co-located with the agent): {'closed', 'items'}."""
    flat = feedback.replace("\n", " ")
    if re.search(r"The %s is closed" % re.escape(r), flat):
        return {"closed": True, "items": None}
    if "you see " not in flat:
        return None
    return {"closed": False, "items": _items_after(flat, "you see ")}


def parse_open(feedback):
    """After 'open r': returns {'opened', 'items'}."""
    flat = feedback.replace("\n", " ")
    m = re.search(r"You open the ([a-z]+ \d+)\.", flat)
    if m is None:
        return None
    return {"opened": m.group(1), "items": _items_after(flat, "you see ") or []}
