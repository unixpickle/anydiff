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
	poolUpstream := p.Pool.Vector.Creator().MakeVector(p.Pool.Vector.Len())
	g[p.Pool] = poolUpstream
	p.Out.Propagate(u, g)
	delete(g, p.Pool)
	p.In.Propagate(poolUpstream, g)
}
