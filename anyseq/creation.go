package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

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
		if u[i].NumPresent() != r.Out[i].NumPresent() {
			panic("invalid present count")
		}
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

// ConstSeqList creates a constant sequence from a list
// of unbatched sequences.
func ConstSeqList(seqs [][]anyvec.Vector) Seq {
	batches := []*Batch{}
	i := 0
	for {
		present := make([]bool, len(seqs))
		var joinMe []anyvec.Vector
		for j, x := range seqs {
			if i >= len(x) {
				continue
			}
			present[j] = true
			joinMe = append(joinMe, x[i])
		}
		if len(joinMe) == 0 {
			break
		}
		batches = append(batches, &Batch{
			Packed:  joinMe[0].Creator().Concat(joinMe...),
			Present: present,
		})
		i++
	}
	return ConstSeq(batches)
}

func (c *constSeq) Output() []*Batch {
	return c.Out
}

func (c *constSeq) Vars() anydiff.VarSet {
	return anydiff.VarSet{}
}

func (c *constSeq) Propagate(u []*Batch, g anydiff.Grad) {
}
