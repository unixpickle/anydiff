package anyseq

import "github.com/unixpickle/anydiff"

// A ResBatch is like a Batch, but it contains a
// differentiable result.
type ResBatch struct {
	Packed  anydiff.Res
	Present []bool
}

type resSeq struct {
	In  []*ResBatch
	Out []*Batch
	V   anydiff.VarSet
}

// ResSeq creates a Seq which propagates itself through
// the result batches.
func ResSeq(b []*ResBatch) Seq {
	out := make([]*Batch, len(b))
	vset := anydiff.VarSet{}
	for i, x := range b {
		out[i] = &Batch{Packed: x.Packed.Output(), Present: x.Present}
		vset = anydiff.MergeVarSets(vset, x.Packed.Vars())
	}
	return &resSeq{In: b, Out: out, V: vset}
}

func (r *resSeq) Output() []*Batch {
	return r.Out
}

func (r *resSeq) Vars() anydiff.VarSet {
	return r.V
}

func (r *resSeq) Propagate(u []*Batch, g anydiff.Grad) {
	for i, x := range r.In {
		x.Packed.Propagate(u[i].Packed, g)
	}
}

type constSeq struct {
	Out []*Batch
}

// ConstSeq creates a batch of sequences from a constant
// list of batches.
func ConstSeq(b []*Batch) Seq {
	return &constSeq{Out: b}
}

func (c *constSeq) Output() []*Batch {
	return c.Out
}

func (c *constSeq) Vars() anydiff.VarSet {
	return anydiff.VarSet{}
}

func (c *constSeq) Propagate(u []*Batch, g anydiff.Grad) {
}
