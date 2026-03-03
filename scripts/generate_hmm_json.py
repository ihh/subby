#!/usr/bin/env python3
"""Generate the Machine Boss JSON for the subby HMM decoder.

Produces a 27-state gene structure HMM (plus Machine Boss end state):
  - 1 intergenic state (IR, shared by both strands)
  - 13 forward-strand gene states
  - 13 reverse-strand gene states

The Tiberius single-strand model has 14 gene states (START + 3E + 3I + 3EI + 3IE + STOP).
To reach exactly 27 states for the dual-strand model, each strand has 13 states:
  - Forward: START, E0, E1, E2, I0, I1, I2, EI0, EI1, EI2, IE0, IE1, IE2
    (STOP is folded into fE1 → IR transition with STOP emission probability)
  - Reverse: STOP, E0, E1, E2, I0, I1, I2, EI0, EI1, EI2, IE0, IE1, IE2
    (START is folded into rE1 → IR transition with START emission probability)

The 15 emission labels consumed from the Mamba tower follow Tiberius:
  0:IR 1:I0 2:I1 3:I2 4:E0 5:E1 6:E2
  7:START 8:EI0 9:EI1 10:EI2 11:IE0 12:IE1 13:IE2 14:STOP
"""

import json
import sys

LABELS = [
    "IR", "I0", "I1", "I2",
    "E0", "E1", "E2",
    "START", "EI0", "EI1", "EI2",
    "IE0", "IE1", "IE2", "STOP",
]


def make_transitions(state_id, targets):
    """Generate transitions over all 15 emission labels.

    Each target is (dest_state, transition_param_name).
    Weight = transition_param * emission_param(label | state).
    Output label = annotation label for this state (strand prefix stripped).
    """
    trans = []
    base = state_id.lstrip("fr")  # strip strand prefix for emission param naming
    for dest, tparam in targets:
        for label in LABELS:
            trans.append({
                "to": dest,
                "in": label,
                "out": base,
                "weight": {"*": [tparam, f"e_{base}_{label}"]},
            })
    return trans


def build_hmm():
    states = []

    # ── IR (intergenic, start state) ──
    states.append({
        "id": "IR",
        "trans": (
            make_transitions("IR", [
                ("IR", "pIR_loop"),
                ("fSTART", "pIR_to_fGene"),
                ("rSTOP", "pIR_to_rGene"),
            ])
            + [{"to": "end", "weight": "pEnd"}]
        ),
    })

    # ── Forward strand (13 states) ──

    # fSTART → fE1
    states.append({
        "id": "fSTART",
        "trans": make_transitions("fSTART", [("fE1", "pfGene_next")]),
    })

    # fE0, fE1, fE2 with phase cycling E0→E1→E2→E0
    for phase in range(3):
        nxt = (phase + 1) % 3
        targets = [
            (f"fE{nxt}", "pfE_next"),
            (f"fEI{phase}", "pfE_to_EI"),
        ]
        if phase == 1:
            # fE1 can end the gene (absorbs STOP function)
            targets.append(("IR", "pfE1_to_IR"))
        states.append({
            "id": f"fE{phase}",
            "trans": make_transitions(f"fE{phase}", targets),
        })

    # fEI0-2 (donor splice) → fI
    for phase in range(3):
        states.append({
            "id": f"fEI{phase}",
            "trans": make_transitions(f"fEI{phase}", [(f"fI{phase}", "pfEI_to_I")]),
        })

    # fI0-2 (intron) self-loop + → fIE
    for phase in range(3):
        states.append({
            "id": f"fI{phase}",
            "trans": make_transitions(f"fI{phase}", [
                (f"fI{phase}", "pfI_loop"),
                (f"fIE{phase}", "pfI_to_IE"),
            ]),
        })

    # fIE0-2 (acceptor splice) → fE
    for phase in range(3):
        states.append({
            "id": f"fIE{phase}",
            "trans": make_transitions(f"fIE{phase}", [(f"fE{phase}", "pfIE_to_E")]),
        })

    # ── Reverse strand (13 states) ──
    # On genome L→R, reverse genes appear: rSTOP → exons/introns → rE1 → IR

    # rSTOP → rE1 (first state encountered for reverse gene)
    states.append({
        "id": "rSTOP",
        "trans": make_transitions("rSTOP", [("rE1", "prGene_next")]),
    })

    # rE0, rE1, rE2 — phase cycling reversed: E0→E2→E1→E0
    for phase in range(3):
        prev = (phase - 1) % 3  # reverse cycling
        targets = [
            (f"rE{prev}", "prE_next"),
            (f"rEI{phase}", "prE_to_EI"),
        ]
        if phase == 1:
            # rE1 can end the reverse gene (absorbs START function)
            targets.append(("IR", "prE1_to_IR"))
        states.append({
            "id": f"rE{phase}",
            "trans": make_transitions(f"rE{phase}", targets),
        })

    # rEI0-2 → rI
    for phase in range(3):
        states.append({
            "id": f"rEI{phase}",
            "trans": make_transitions(f"rEI{phase}", [(f"rI{phase}", "prEI_to_I")]),
        })

    # rI0-2 self-loop + → rIE
    for phase in range(3):
        states.append({
            "id": f"rI{phase}",
            "trans": make_transitions(f"rI{phase}", [
                (f"rI{phase}", "prI_loop"),
                (f"rIE{phase}", "prI_to_IE"),
            ]),
        })

    # rIE0-2 → rE
    for phase in range(3):
        states.append({
            "id": f"rIE{phase}",
            "trans": make_transitions(f"rIE{phase}", [(f"rE{phase}", "prIE_to_E")]),
        })

    # ── End state ──
    states.append({"id": "end"})

    return {"state": states}


def main():
    hmm = build_hmm()
    out_path = "src/model/hmm_decoder.json"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    with open(out_path, "w") as f:
        json.dump(hmm, f, indent=2)

    n_states = len(hmm["state"])
    n_gene_states = n_states - 1  # exclude 'end'
    n_trans = sum(len(s.get("trans", [])) for s in hmm["state"])
    print(f"States: {n_gene_states} gene-structure + 1 end = {n_states} total")
    print(f"Transitions: {n_trans}")
    print(f"Emission labels: {len(LABELS)}")
    print(f"Wrote to {out_path}")


if __name__ == "__main__":
    main()
