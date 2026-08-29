# Ground-truth causal links for:
# A1 / Gaussian / 4 variables / Lag 2

TRUE_LINKS = {
    ("X1", -2, "X4"): 0.25,
    ("X4",  0, "X3"): 0.35,
    ("X3", -1, "X2"): 0.30,
    ("X2",  0, "X1"): 0.40,
}

print("Ground-truth causal links:")
print()

for (source, lag, target), weight in TRUE_LINKS.items():
    print(f"{source} --(lag={lag})--> {target}   weight={weight}")
