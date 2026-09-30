class Scheduler:
    """Build per-stage gate groups from a flat gate list ``g_q`` (indices into ``g_q``)."""

    def __init__(self,
                 g_q: list,
                 results_code: dict,
                 barriers: list = None,
                 ):
        """
        Args:
            g_q: Flat list of gates as ``(q0, q1)`` with ``q0==q1`` for 1-qubit gates.
            results_code: Mutated in place; ``stages`` is filled by ``save_results``.
            barriers: Optional list of ``(pos, qubits)``. The barrier sits just before
                ``g_q[pos]`` in program order (``pos == len(g_q)`` allowed); ``qubits`` is a
                tuple of qubit ids, empty meaning all qubits. A barrier is a dependency node
                scoped to its qubits: it waits for the latest earlier gate on each of them, and
                every later gate touching any of them waits for it. It occupies no stage and is
                not emitted to ``results_code``.
        """
        self.g_q = g_q
        self.results_code = results_code
        self.results_code['stages'] = {
            "num_stage": 0,
            "stage": {q: {} for q in range(self.results_code['n_q'])},
            "qs_status": {}
        }
        n_q = self.results_code['n_q']
        self.barriers = []
        for pos, qubits in sorted(barriers or [], key=lambda b: b[0]):
            if not 0 <= pos <= len(g_q):
                raise ValueError(f"barrier position {pos} outside [0, {len(g_q)}]")
            if pos in (0, len(g_q)):
                continue  # nothing before it or nothing after it: constrains no pair of gates
            self.barriers.append((pos, tuple(qubits) or tuple(range(n_q))))
        self.list_scheduling = []

    def _barriers_at(self, i):
        return [qs for pos, qs in self.barriers if pos == i]

    def asap_joint(self):
        """ASAP over the full gate stream: each gate starts when both qubits are free."""

        list_qubit_stage = [0 for _ in range(self.results_code['n_q'])]
        for i, gate in enumerate(self.g_q):
            for qs in self._barriers_at(i):
                s = max(list_qubit_stage[q] for q in qs)
                for q in qs:
                    list_qubit_stage[q] = s
            stage0 = list_qubit_stage[gate[0]]
            stage1 = list_qubit_stage[gate[1]]
            stage = max(stage0, stage1)
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(i)

            stage += 1
            list_qubit_stage[gate[0]] = stage
            list_qubit_stage[gate[1]] = stage
        self.save_results()

    def asap_separate(self):
        """ASAP schedule for all 2q layers first, then place 1q gates without reordering 2q stages."""
        if self.barriers:
            self._asap_separate_barriers()
            self.save_results()
            return
        list_qubit_stage = [0 for _ in range(self.results_code['n_q'])]

        two_qubit_gates = []
        single_qubit_gates = []

        for i, gate in enumerate(self.g_q):
            if gate[0] == gate[1]:
                single_qubit_gates.append((i, gate))
            else:
                two_qubit_gates.append((i, gate))

        for i, gate in two_qubit_gates:
            stage0 = list_qubit_stage[gate[0]]
            stage1 = list_qubit_stage[gate[1]]
            stage = max(stage0, stage1)
            if stage >= len(self.list_scheduling):
                self.list_scheduling.append([])
            self.list_scheduling[stage].append(i)

            stage += 1
            list_qubit_stage[gate[0]] = stage
            list_qubit_stage[gate[1]] = stage

        prev_2q_stage = {}
        for i, gate in enumerate(self.g_q):
            if i in [idx for sublist in self.list_scheduling for idx in sublist]:
                prev_2q_stage[gate[0]] = next(idx for idx, sublist in enumerate(self.list_scheduling) if i in sublist)
                prev_2q_stage[gate[1]] = next(idx for idx, sublist in enumerate(self.list_scheduling) if i in sublist)
            else:
                stage = max(prev_2q_stage.get(gate[0], -1), prev_2q_stage.get(gate[1], -1)) + 1

                while stage >= len(self.list_scheduling):
                    self.list_scheduling.append([])

                existing_two_qubit_gates = any(
                    self.g_q[j][0] != self.g_q[j][1] for j in self.list_scheduling[stage]
                )

                if existing_two_qubit_gates:
                    self.list_scheduling.insert(stage, [i])
                else:
                    self.list_scheduling[stage].append(i)

        self.save_results()

    def _asap_separate_barriers(self):
        """Barrier-aware variant of ``asap_separate``.

        Every gate gets an integer position: even ``2k`` is the 1q slot before 2q layer ``k``,
        odd ``2k+1`` is 2q layer ``k``. Gates are placed in one program-order pass (1q and 2q
        together), so barriers constrain both kinds in both directions. Positions are then
        compacted into stages. Unlike the legacy path, no stages are inserted after the fact.
        """
        n_q = self.results_code['n_q']
        ready1 = [0] * n_q  # earliest position for the next 1q gate on q
        ready2 = [1] * n_q  # earliest position for the next 2q gate on q
        last = [-1] * n_q   # position of the latest gate on q
        placed = {}
        for i, (a, b) in enumerate(self.g_q):
            for qs in self._barriers_at(i):
                floor = max(last[q] for q in qs) + 1
                for q in qs:
                    ready1[q] = max(ready1[q], floor)
                    ready2[q] = max(ready2[q], floor)
            if a == b:
                p = ready1[a] + ready1[a] % 2
                ready1[a], ready2[a], last[a] = p, p + 1, p
            else:
                p = max(ready2[a], ready2[b])
                p += 1 - p % 2
                for q in (a, b):
                    ready1[q], ready2[q], last[q] = p + 1, p + 2, p
            placed.setdefault(p, []).append(i)
        self.list_scheduling = [placed[p] for p in sorted(placed)]

    def segment_starts(self):
        """Stage indices where a new execution segment begins, cut at each full-circuit barrier.

        A full barrier orders every gate before it strictly ahead of every gate after it, so no
        stage straddles a cut. Call after ``asap_*``. Stage 0 is not listed; consecutive full
        barriers with no gates between them yield one cut.
        """
        n_q = self.results_code['n_q']
        full = [pos for pos, qs in self.barriers if len(set(qs)) == n_q]
        seg = lambda i: sum(pos <= i for pos in full)
        starts = []
        prev = 0
        for stage, gates in enumerate(self.list_scheduling):
            ids = {seg(i) for i in gates}
            assert len(ids) == 1, f"stage {stage} straddles a full barrier"
            (cur,) = ids
            if cur != prev:
                starts.append(stage)
                prev = cur
        return starts

    def save_results(self):
        """Write ``list_scheduling`` into ``results_code['stages']`` for the placer/router."""
        stage_dict = {}
        for stage, gates in enumerate(self.list_scheduling):
            if all(self.g_q[gate][0] == self.g_q[gate][1] for gate in gates):
                stage_type = "1qGate"
            elif all(self.g_q[gate][0] != self.g_q[gate][1] for gate in gates):
                stage_type = "2qGate"
            else:
                stage_type = "mGate"
            stage_dict[stage] = {
                'type': stage_type,
                'idx': gates,
                'gates': [self.g_q[gate] for gate in gates]
            }

        qs_status = {q: [{"stage": stage_id, "status": None} for stage_id in range(len(self.list_scheduling))] for q in range(self.results_code['n_q'])}
        for stage, gates in enumerate(self.list_scheduling):
            for gate in gates:
                q0, q1 = self.g_q[gate]
                if q0 == q1:
                    qs_status[q0][stage]['status'] = "1qGate"
                else:
                    qs_status[q0][stage]['status'] = "2qGate"
                    qs_status[q1][stage]['status'] = "2qGate"

        self.results_code['stages']['qs_status'] = qs_status
        self.results_code['stages']['num_stage'] = len(self.list_scheduling)
        self.results_code['stages']['stage'] = stage_dict
