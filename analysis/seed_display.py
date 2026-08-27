GROUP_LABELS = {
    "lucky": "Lucky",
    "moderate": "Moderate",
    "ordinary": "Ordinary",
}

GROUP_COLORS = {
    "Lucky": "#dc2626",
    "Moderate": "#f59e0b",
    "Ordinary": "#2563eb",
}


def label(group: str) -> str:
    if group not in GROUP_LABELS:
        raise ValueError(f"Unknown seed group: {group}")
    return GROUP_LABELS[group]
