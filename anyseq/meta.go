package anyseq

import "github.com/unixpickle/anydiff"

// Pool calls f with a copy of s in such a way that s will
// only be back-propagated through once.
func Pool(s Seq, f func(s Seq) Seq) Seq {
	var pool []*anydiff.Var
	var resBatches []*ResBatch
	for _, x := range s.Output() {
		p := anydiff.NewVar(x.Packed)
		pool = append(pool, p)
		resBatches = append(resBatches, &ResBatch{Packed: p, Present: x.Present})
	}
	pooledIn := ResSeq(resBatches)
	out := f(pooledIn)

	vars := anydiff.MergeVarSets(out.Vars(), s.Vars())
	for _, p := range pool {
		vars.Del(p)
	}

	return &poolRes{
		In:   s,
		Pool: pool,
		Res:  out,
		V:    vars,
	}
}

type poolRes struct {
	In   Seq
	Pool []*anydiff.Var
	Res  Seq
	V    anydiff.VarSet
}

func (p *poolRes) Output() []*Batch {
	return p.Res.Output()
}

func (p *poolRes) Vars() anydiff.VarSet {
	return p.V
}

func (p *poolRes) Propagate(u []*Batch, grad anydiff.Grad) {
	for _, x := range p.Pool {
		grad[x] = x.Vector.Creator().MakeVector(x.Vector.Len())
	}
	p.Res.Propagate(u, grad)
	var downstream []*Batch
	for i, x := range p.Pool {
		downstream = append(downstream, &Batch{
			Packed:  grad[x],
			Present: u[i].Present,
		})
		delete(grad, x)
	}
	p.In.Propagate(downstream, grad)
}
