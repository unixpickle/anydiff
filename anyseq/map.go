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
// It is guaranteed that f will be called for each
// timestep in order.
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

type mapNResult struct {
	In   []Seq
	Pool [][]*anydiff.Var
	Res  []anydiff.Res
	Out  []*Batch
	V    anydiff.VarSet
}

// MapN maps a batched function over each timestep of the
// sequences in a differentiable manner.
//
// All the sequences must have the same shape.
//
// It is guaranteed that f will be called for each
// timestep in order.
//
// For each output, f is passed a one anydiff.Res for each
// sequence, containing the packed batch contents.
// It is also passed the batch size, which must match
// across all the sequences.
//
// The result of f must have a length divisible by n,
// since said result is treated as a packed batch of size
// n.
func MapN(f func(n int, v ...anydiff.Res) anydiff.Res, s ...Seq) Seq {
	if len(s) == 0 {
		panic("must take at least one sequence")
	}
	pool := make([][]*anydiff.Var, len(s))
	res := make([]anydiff.Res, len(s[0].Output()))
	out := make([]*Batch, len(s[0].Output()))
	allVars := anydiff.VarSet{}
	for i, seq := range s {
		if len(seq.Output()) != len(out) {
			panic("input shape mismatch")
		}
		allVars = anydiff.MergeVarSets(allVars, seq.Vars())
		pool[i] = make([]*anydiff.Var, len(seq.Output()))
	}
	for i := range out {
		var reses []anydiff.Res
		n := s[0].Output()[i].NumPresent()
		present := s[0].Output()[i].Present
		for j, seq := range s {
			out := seq.Output()[i]
			pool[j][i] = anydiff.NewVar(out.Packed)
			if out.NumPresent() != n {
				panic("input shape mismatch")
			}
			reses = append(reses, pool[j][i])
		}
		res[i] = f(n, reses...)
		if res[i].Output().Len()%n != 0 {
			panic("mapped function must give a batch of results")
		}
		out[i] = &Batch{Packed: res[i].Output(), Present: present}
		allVars = anydiff.MergeVarSets(allVars, res[i].Vars())
	}
	for _, x := range pool {
		for _, y := range x {
			allVars.Del(y)
		}
	}
	return &mapNResult{
		In:   s,
		Pool: pool,
		Res:  res,
		Out:  out,
		V:    allVars,
	}
}

func (m *mapNResult) Output() []*Batch {
	return m.Out
}

func (m *mapNResult) Vars() anydiff.VarSet {
	return m.V
}

func (m *mapNResult) Propagate(u []*Batch, g anydiff.Grad) {
	downstream := make([][]*Batch, len(m.In))
	for i, inSeq := range m.In {
		if g.Intersects(inSeq.Vars()) {
			downstream[i] = make([]*Batch, len(m.Pool[i]))
			for j, x := range m.Pool[i] {
				d := x.Vector.Creator().MakeVector(x.Vector.Len())
				g[x] = d
				downstream[i][j] = &Batch{Packed: d, Present: u[i].Present}
			}
		}
	}

	for i, o := range m.Res {
		o.Propagate(u[i].Packed, g)
	}

	for i, down := range downstream {
		if down != nil {
			for _, x := range m.Pool[i] {
				delete(g, x)
			}
		}
	}
	for i, down := range downstream {
		if down != nil {
			m.In[i].Propagate(down, g)
		}
	}
}
