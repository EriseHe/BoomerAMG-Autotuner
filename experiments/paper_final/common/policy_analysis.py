"""Execution-cell identities used by matched-policy workers."""


def cell_key(row):
    return (
        row["seed"],
        row["source"],
        row["case_id"],
        row.get("repeat", 0),
        row["policy"],
    )


def expected_cells(jobs):
    keys = [cell_key({**job, "policy": name}) for job in jobs for name in job["policy_order"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate planned execution")
    return set(keys)
