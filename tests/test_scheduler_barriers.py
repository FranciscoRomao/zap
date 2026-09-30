import random

import pytest

from zap.scheduler.scheduler import Scheduler
from tests._old_scheduler import Scheduler as OldScheduler

STRATEGIES = ["asap_joint", "asap_separate"]


def run(cls, g_q, n_q, strategy, **kw):
    rc = {'n_q': n_q}
    s = cls(g_q, rc, **kw)
    getattr(s, strategy)()
    return s.list_scheduling, rc['stages']


def random_gates(n_q, n, rng):
    gates = []
    for _ in range(n):
        a, b = rng.randrange(n_q), rng.randrange(n_q)
        gates.append((a, b if rng.random() < 0.6 else a))
    return gates


def stage_map(sched):
    return {i: s for s, gs in enumerate(sched) for i in gs}


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("barriers", [None, []])
@pytest.mark.parametrize("seed", range(10))
def test_default_matches_old(strategy, barriers, seed):
    g_q = random_gates(6, 80, random.Random(seed))
    assert run(Scheduler, g_q, 6, strategy, barriers=barriers) == run(OldScheduler, g_q, 6, strategy)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_barrier_scoping(strategy):
    g_q = [(0, 1), (2, 3)]
    free, _ = run(Scheduler, g_q, 4, strategy)
    assert free == [[0, 1]]
    # barrier links qubit 1 (first pair) and 2 (second pair): gate 1 must wait for gate 0
    linked, stages = run(Scheduler, g_q, 4, strategy, barriers=[(1, (1, 2))])
    assert linked == [[0], [1]]
    assert stages['num_stage'] == 2 and stages['stage'][1]['idx'] == [1]
    # barrier on qubits touched by neither gate constrains nothing
    g_q = [(0, 1), (2, 3), (4, 5)]
    assert run(Scheduler, g_q, 6, strategy, barriers=[(2, (4, 5))])[0] == [[0, 1, 2]]
    # barrier on only one side: gates after it on other qubits stay free
    assert run(Scheduler, g_q, 6, strategy, barriers=[(1, (0, 1))])[0] == [[0, 1, 2]]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("seed", range(40))
def test_barrier_property(strategy, seed):
    rng = random.Random(seed)
    n_q = 5
    g_q = random_gates(n_q, 40, rng)
    barriers = []
    for _ in range(rng.randrange(1, 5)):
        k = rng.randrange(0, n_q + 1)
        qs = tuple(sorted(rng.sample(range(n_q), k)))
        barriers.append((rng.randrange(1, len(g_q)), qs))
    sched, stages = run(Scheduler, g_q, n_q, strategy, barriers=barriers)
    st = stage_map(sched)
    assert sorted(st) == list(range(len(g_q)))  # each gate exactly once; no barrier nodes
    assert stages['num_stage'] == len(sched)
    assert all(g in range(len(g_q)) for gs in sched for g in gs)

    # original per-qubit program order preserved (consecutive 1q gates may share a stage)
    last = {}
    for i, g in enumerate(g_q):
        for q in set(g):
            if q in last:
                j = last[q]
                both_1q = g[0] == g[1] and g_q[j][0] == g_q[j][1]
                assert st[j] <= st[i] if both_1q else st[j] < st[i]
            last[q] = i

    # barrier ordering
    for pos, qs in barriers:
        qs = set(qs) or set(range(n_q))
        for e in range(pos):
            if not qs & set(g_q[e]):
                continue
            for l in range(pos, len(g_q)):
                if qs & set(g_q[l]):
                    assert st[e] < st[l], (pos, qs, e, l)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_noop_barriers(strategy):
    g_q = random_gates(5, 40, random.Random(7))
    old = run(OldScheduler, g_q, 5, strategy)
    # leading barrier has nothing before it; trailing barrier nothing after it
    for barriers in ([(0, ())], [(len(g_q), ())], [(0, ()), (len(g_q), (1, 2))]):
        assert run(Scheduler, g_q, 5, strategy, barriers=barriers) == old


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("seed", range(30))
def test_segment_starts(strategy, seed):
    rng = random.Random(seed)
    n_q = 5
    g_q = random_gates(n_q, 40, rng)
    full_pos = sorted(rng.sample(range(1, len(g_q)), rng.randrange(0, 4)))
    barriers = [(p, ()) for p in full_pos] + [(rng.randrange(1, len(g_q)), (0, 1))]
    rc = {'n_q': n_q}
    s = Scheduler(g_q, rc, barriers=barriers)
    getattr(s, strategy)()
    starts = s.segment_starts()
    bounds = [0] + starts + [len(s.list_scheduling)]
    segs = [[i for st in s.list_scheduling[a:b] for i in st] for a, b in zip(bounds, bounds[1:])]
    assert sorted(i for seg in segs for i in seg) == list(range(len(g_q)))
    cuts = [0] + full_pos + [len(g_q)]
    expected = [set(range(a, b)) for a, b in zip(cuts, cuts[1:])]
    assert [set(seg) for seg in segs] == expected
