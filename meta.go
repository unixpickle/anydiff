package anydiff

import "github.com/unixpickle/anyvec"

type poolRes struct {
	Pool *Var
	V    VarSet
	In   Res
	Out  Res
}

// Pool calls f in such a way that f sees and can use the
// result r without back-propagating through r more than
// once.
//
// If you plan on using a Res more than once to compute
// another result, using Pool can prevent that Res from
// being propagated through multiple times.
func Pool(r Res, f func(r Res) Res) Res {
	switch r := r.(type) {
	case *Var:
		return f(r)
	case *Const:
		return f(r)
	}
	pool := NewVar(r.Output())
	out := f(pool)
	if !out.Vars().Has(pool) {
		return out
	}
	newVars := MergeVarSets(out.Vars(), r.Vars())
	newVars.Del(pool)
	return &poolRes{
		Pool: pool,
		V:    newVars,
		In:   r,
		Out:  out,
	}
}

func (p *poolRes) Output() anyvec.Vector {
	return p.Out.Output()
}

func (p *poolRes) Vars() VarSet {
	return p.V
}

func (p *poolRes) Propagate(u anyvec.Vector, g Grad) {
	propIn := g.Intersects(p.In.Vars())
	if propIn {
		g[p.Pool] = p.Pool.Vector.Creator().MakeVector(p.Pool.Vector.Len())
	}
	p.Out.Propagate(u, g)
	if propIn {
		down := g[p.Pool]
		delete(g, p.Pool)
		p.In.Propagate(down, g)
	}
}

type poolMultiRes struct {
	In    MultiRes
	Out   MultiRes
	Pools []*Var
	V     VarSet
}

// PoolMulti splits m into separate Res objects, passes
// the result to f, and echos the result of f in such a
// way that m is only propagated through once.
func PoolMulti(m MultiRes, f func(reses []Res) MultiRes) MultiRes {
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
	return &poolMultiRes{
		In:    m,
		Out:   out,
		Pools: pool,
		V:     vars,
	}
}

func (p *poolMultiRes) Outputs() []anyvec.Vector {
	return p.Out.Outputs()
}

func (p *poolMultiRes) Vars() VarSet {
	return p.V
}

func (p *poolMultiRes) Propagate(u []anyvec.Vector, g Grad) {
	propIn := g.Intersects(p.In.Vars())
	if propIn {
		for _, x := range p.Pools {
			g[x] = x.Vector.Creator().MakeVector(x.Vector.Len())
		}
	}

	p.Out.Propagate(u, g)

	if propIn {
		down := make([]anyvec.Vector, len(p.Pools))
		for i, x := range p.Pools {
			down[i] = g[x]
			delete(g, x)
		}
		p.In.Propagate(down, g)
	}
}
