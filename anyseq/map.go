package anyseq

import "github.com/unixpickle/anydiff"

type mapResult struct {
	In   Seq
	Pool []*anydiff.Var
	Res  []anydiff.Res
	Out  []*Batch
	V    anydiff.VarSet
}

// Map maps a batched function over each timestep of the
// sequence in a differentiable manner.
//
// For each output, f is passed an anydiff.Res containing
// the packed batch contents, as well as the batch size.
// The result of f must be divisible by n, since said
// result is treated as a packed batch of size n.
func Map(s Seq, f func(v anydiff.Res, n int) anydiff.Res) Seq {
	pool := make([]*anydiff.Var, len(s.Output()))
	res := make([]anydiff.Res, len(s.Output()))
	out := make([]*Batch, len(s.Output()))
	allVars := anydiff.MergeVarSets(s.Vars())
	for i, x := range s.Output() {
		pool[i] = anydiff.NewVar(x.Packed)
		n := x.NumPresent()
		res[i] = f(pool[i], n)
		if res[i].Output().Len()%n != 0 {
			panic("mapped function must give a batch of results")
		}
		out[i] = &Batch{Packed: res[i].Output(), Present: x.Present}
		allVars = anydiff.MergeVarSets(allVars, res[i].Vars())
	}
	for _, x := range pool {
		allVars.Del(x)
	}
	return &mapResult{
		In:   s,
		Pool: pool,
		Res:  res,
		Out:  out,
		V:    allVars,
	}
}

func (m *mapResult) Output() []*Batch {
	return m.Out
}

func (m *mapResult) Vars() anydiff.VarSet {
	return m.V
}

func (m *mapResult) Propagate(u []*Batch, g anydiff.Grad) {
	if !g.Intersects(m.In.Vars()) {
		for i, o := range m.Res {
			o.Propagate(u[i].Packed, g)
		}
		return
	}

	downstream := make([]*Batch, len(m.Out))
	for i, x := range m.Pool {
		d := x.Vector.Creator().MakeVector(x.Vector.Len())
		g[x] = d
		downstream[i] = &Batch{Packed: d, Present: u[i].Present}
	}
	for i, o := range m.Res {
		o.Propagate(u[i].Packed, g)
	}
	for _, x := range m.Pool {
		delete(g, x)
	}
	m.In.Propagate(downstream, g)
}
