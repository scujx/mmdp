"""Static topology-only generation; imports no environment or training code."""

from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parent / "configs"
SOURCE = Path(__file__).resolve().parent / "configs/environment.json"


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reach(start, edges, directed):
    found = {start}
    while True:
        added = {v for u, v in edges if u in found}
        if not directed:
            added |= {u for u, v in edges if v in found}
        grown = found | added
        if grown == found:
            return found
        found = grown


def main():
    assert not (ROOT / "mask_inventory.json").exists()
    assert np.__version__ == "2.0.1"
    design = ROOT / "design.md"
    assert design.exists()
    cfg = json.loads(SOURCE.read_text())
    routes = tuple(tuple(x) for x in cfg["stage1_edges"])
    persistent = tuple(tuple(x) for x in cfg["stage2_edges"])
    true = tuple(i for i, e in enumerate(routes) if e not in persistent)
    roles = {k: cfg[k] for k in ["master_idx", "investment_idx", "operational_indices"]}
    assert cfg["num_accounts"] == 5 and roles == dict(
        master_idx=0, investment_idx=1, operational_indices=[2, 3, 4]
    )
    assert (
        len(routes) == len(set(routes)) == 14
        and len(persistent) == 8
        and len(true) == 6
    )
    assert set(persistent) <= set(routes)
    mandatory = tuple(
        routes.index((roles["master_idx"], i)) for i in roles["operational_indices"]
    )
    assert len(mandatory) == 3 and set(mandatory) <= set(true)
    rest = tuple(i for i in range(14) if i not in mandatory)
    cpool = sorted(
        tuple(sorted(mandatory + extra))
        for extra in itertools.combinations(rest, 3)
        if tuple(sorted(mandatory + extra)) != true
    )
    assert len(cpool) == len(set(cpool)) == 164
    settings = []

    def record(name, typ, indices, seed=None, draws=None):
        edges = [routes[i] for i in indices]
        nodes = sorted({n for e in edges for n in e})
        components = []
        unseen = set(range(5))
        while unseen:
            comp = reach(min(unseen), edges, False)
            components.append(sorted(comp))
            unseen -= comp
        covered = [
            i for i in roles["operational_indices"] if (roles["master_idx"], i) in edges
        ]
        diagnostics = dict(
            true_overlap=len(set(indices) & set(true)),
            expiring_count=sum(i in true for i in indices),
            persistent_count=sum(i not in true for i in indices),
            direct_funding_covered=covered,
            direct_funding_count=len(covered),
            nodes_covered=nodes,
            isolated_nodes=sorted(set(range(5)) - set(nodes)),
            weak_components=components,
            weakly_connected=len(components) == 1,
            strongly_connected=all(len(reach(n, edges, True)) == 5 for n in range(5)),
        )
        settings.append(
            dict(
                setting_id=name,
                mask_type=typ,
                stage0_indices=list(indices),
                stage0_routes=edges,
                design_rng_seed=seed,
                draw_history=draws or [],
                diagnostics=diagnostics,
            )
        )

    record("Flat", "flat", tuple(range(14)))
    record("True", "true", true)
    chosen_c = []
    for k in range(1, 6):
        rng = np.random.Generator(np.random.PCG64(982200 + k))
        draws = []
        while True:
            ix = int(rng.integers(len(cpool)))
            candidate = cpool[ix]
            accept = candidate not in chosen_c
            draws.append(
                dict(
                    candidate_index=ix,
                    stage0_indices=candidate,
                    accepted=accept,
                    reason=(
                        "first unused candidate" if accept else "duplicate of prior C"
                    ),
                )
            )
            if accept:
                break
        chosen_c.append(candidate)
        record(f"C{k}", "conditional", candidate, 982200 + k, draws)
    upool = [
        x
        for x in itertools.combinations(range(14), 6)
        if x != true and x not in chosen_c
    ]
    assert len(upool) == 2997
    chosen_u = []
    for k in range(1, 6):
        rng = np.random.Generator(np.random.PCG64(982100 + k))
        draws = []
        while True:
            ix = int(rng.integers(len(upool)))
            candidate = upool[ix]
            accept = candidate not in chosen_u
            draws.append(
                dict(
                    candidate_index=ix,
                    stage0_indices=candidate,
                    accepted=accept,
                    reason=(
                        "first unused candidate" if accept else "duplicate of prior U"
                    ),
                )
            )
            if accept:
                break
        chosen_u.append(candidate)
        record(f"U{k}", "uniform", candidate, 982100 + k, draws)
    assert len(set(chosen_c + chosen_u)) == 10 and true not in chosen_c + chosen_u
    order = (
        ["Flat", "True"]
        + [f"U{k}" for k in range(1, 6)]
        + [f"C{k}" for k in range(1, 6)]
    )
    settings.sort(key=lambda x: order.index(x["setting_id"]))
    obj = dict(
        schema_version=1,
        status="GENERATED_ONCE_TOPOLOGY_ONLY",
        catalogue_routes=routes,
        persistent_routes=persistent,
        true_routes=[routes[i] for i in true],
        roles=roles,
        settings=settings,
        numpy_version=np.__version__,
        rng="PCG64",
        source_json=str(SOURCE),
        source_json_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        design_sha256_before_generation=hashlib.sha256(design.read_bytes()).hexdigest(),
        generation_order="C1..C5 then U1..U5; independent PCG64 streams, reject prior-within-group duplicates only",
        candidate_pools=dict(
            C=dict(size=164, sha256=digest(cpool)),
            U=dict(
                size=2997,
                sha256=digest(upool),
                excluded_true=True,
                excluded_selected_C_masks=chosen_c,
            ),
        ),
        environment_instances_constructed=0,
        returns_read=0,
    )
    (ROOT / "mask_inventory.json").write_text(json.dumps(obj, indent=2) + "\n")
    (ROOT / "mask_inventory.json.sha256").write_text(
        hashlib.sha256((ROOT / "mask_inventory.json").read_bytes()).hexdigest()
        + "  mask_inventory.json\n"
    )
    print(
        json.dumps({r["setting_id"]: r["stage0_indices"] for r in settings}, indent=2)
    )


if __name__ == "__main__":
    main()
