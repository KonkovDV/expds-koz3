from evlib import *
cv = curves_of({c: stack_preds(c) for c in GEO})
cs, info = counts_with(cv)
for c in GEO:
    a = M.score_counts({c: cs[c]})[1][c]; b = M.score_counts({c: BASE['counts'][c]})[1][c]
    print(c, 'repro', {k: a[k] for k in ('tp','fp','n_gt')}, 'base', {k: b[k] for k in ('tp','fp','n_gt')}, 'same', cs[c] == BASE['counts'][c])
print('scores', scores(cs), 'base', BASE['scores'], info)
