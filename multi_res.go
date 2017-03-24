package anydiff

import "github.com/unixpickle/anyvec"

// MultiRes is a collection of results which can be
// propagated through simultaneously.
//
// This type is designed to be useful in conjunction with
// the PoolMulti API.
type MultiRes interface {
	// Outputs returns the output for each result.
	Outputs() []anyvec.Vector

	// Vars returns the variables upon which the results
	// depend.
	Vars() VarSet

	// Propagate propagates through all of the outputs
	// at once.
	//
	// This may modify the upstream vectors.
	Propagate(upstream []anyvec.Vector, grad Grad)
}

type fuseRes struct {
	Ins  []Res
	Outs []anyvec.Vector
	V    VarSet
}

// Fuse creates a MultiRes from zero or more Reses.
func Fuse(reses ...Res) MultiRes {
	res := &fuseRes{V: VarSet{}}
	for _, x := range reses {
		res.Ins = append(res.Ins, x)
		res.Outs = append(res.Outs, x.Output())
		res.V = MergeVarSets(res.V, x.Vars())
	}
	return res
}

func (f *fuseRes) Outputs() []anyvec.Vector {
	return f.Outs
}

func (f *fuseRes) Vars() VarSet {
	return f.V
}

func (f *fuseRes) Propagate(u []anyvec.Vector, g Grad) {
	for i, x := range f.Ins {
		x.Propagate(u[i], g)
	}
}

// FuseMulti fuses together the results of multiple
// MultiRes instances.
func FuseMulti(m ...MultiRes) MultiRes {
	if len(m) == 0 {
		return Fuse()
	} else if len(m) == 1 {
		return m[0]
	}
	m1 := FuseMulti(m[:len(m)/2]...)
	m2 := FuseMulti(m[len(m)/2:]...)
	return PoolMulti(m1, func(m1Res []Res) MultiRes {
		return PoolMulti(m2, func(m2Res []Res) MultiRes {
			return Fuse(append(append([]Res{}, m1Res...), m2Res...)...)
		})
	})
}

type unfuseRes struct {
	In    MultiRes
	Out   Res
	Pools []*Var
	V     VarSet
}

// Unfuse separates the results in m, calls f with the
// separated results, and echos the result of f in such a
// way that m will only be propagated through once.
func Unfuse(m MultiRes, f func(reses []Res) Res) Res {
	var pool []*Var
	var reses []Res
	for _, x := range m.Outputs() {
		p := NewVar(x)
		pool = append(pool, p)
		reses = append(reses, p)
	}
	out := f(reses)
	vars := MergeVarSets(out.Vars(), m.Vars())
	for _, x := range pool {
		vars.Del(x)
	}
	return &unfuseRes{
		In:    m,
		Out:   out,
		Pools: pool,
		V:     vars,
	}
}

func (u *unfuseRes) Output() anyvec.Vector {
	return u.Out.Output()
}

func (u *unfuseRes) Vars() VarSet {
	return u.V
}

func (u *unfuseRes) Propagate(up anyvec.Vector, g Grad) {
	propIn := g.Intersects(u.In.Vars())
	if propIn {
		for _, x := range u.Pools {
			g[x] = x.Vector.Creator().MakeVector(x.Vector.Len())
		}
	}
	u.Out.Propagate(up, g)
	if propIn {
		down := make([]anyvec.Vector, len(u.Pools))
		for i, x := range u.Pools {
			down[i] = g[x]
			delete(g, x)
		}
		u.In.Propagate(down, g)
	}
}
