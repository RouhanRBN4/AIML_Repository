# ------------------------
# Forward Chaining
# ------------------------

def forward_chaining(facts, rules):
    inferred = set(facts)
    added = True

    while added:
        added = False
        for premise, conclusion in rules:
            if premise.issubset(inferred) and conclusion not in inferred:
                inferred.add(conclusion)
                added = True

    return inferred

# ------------------------
# Backward Chaining
# ------------------------

def backward_chaining(goal, facts, rules):
    if goal in facts:
        return True

    for premise, conclusion in rules:
        if conclusion == goal:
            if all(backward_chaining(p, facts, rules) for p in premise):
                return True

    return False

# ------------------------
# Resolution Strategy
# ------------------------

def resolve(c1, c2):
    resolvents = []
    for literal in c1:
        if ('-' + literal) in c2:
            new_clause = (c1 - {literal}) | (c2 - {'-' + literal})
            resolvents.append(new_clause)
        elif literal.startswith('-') and literal[1:] in c2:
            new_clause = (c1 - {literal}) | (c2 - {literal[1:]})
            resolvents.append(new_clause)
    return resolvents

def resolution(kb, query):
    kb = [set(clause) for clause in kb]
    kb.append({'-' + query})

    while True:
        new = []

        for i in range(len(kb)):
            for j in range(i + 1, len(kb)):
                resolvents = resolve(kb[i], kb[j])

                for r in resolvents:
                    if r == set():
                        return True
                    new.append(r)

        if all(r in kb for r in new):
            return False

        for r in new:
            if r not in kb:
                kb.append(r)

# ------------------------
# Knowledge Base
# ------------------------

facts = {"A"}

rules = [
    ({"A"}, "B"),
    ({"B"}, "C"),
    ({"C"}, "D")
]

goal = "D"

kb = [
    {"A", "B"},
    {"-A"}
]

query = "B"

# ------------------------
# Results
# ------------------------

fc_result = forward_chaining(facts, rules)
bc_result = backward_chaining(goal, facts, rules)
res_result = resolution(kb, query)

print("Forward Chaining Result:", fc_result)
print("Backward Chaining Result:", bc_result)
print("Resolution Result:", res_result)