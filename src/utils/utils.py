from collections import defaultdict


def list2dic(data: list[dict]) -> dict[str, list]:
    ret = defaultdict(list)
    for item in data:
        for k, v in item.items():
            ret[k].append(v)
    return ret