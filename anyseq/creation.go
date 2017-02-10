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
	C   anyvec.Creator
	In  []*ResBatch
	Out []*Batch
	V   anydiff.VarSet
}

// ResSeq creates a Seq which propagates itself through
// the result batches.
func ResSeq(c anyvec.Creator, b []*ResBatch) Seq {
	out := make([]*Batch, len(b))
	vset := anydiff.VarSet{}
	for i, x := range b {
		out[i] = &Batch{Packed: x.Packed.Output(), Present: x.Present}
		vset = anydiff.MergeVarSets(vset, x.Packed.Vars())
	}
	return &resSeq{C: c, In: b, Out: out, V: vset}
}

func (r *resSeq) Creator() anyvec.Creator {
	return r.C
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
	C   anyvec.Creator
	Out []*Batch
}

// ConstSeq creates a batch of sequences from a constant
// list of batches.
func ConstSeq(c anyvec.Creator, b []*Batch) Seq {
	return &constSeq{C: c, Out: b}
}

// ConstSeqList creates a constant sequence from a list
// of unbatched sequences.
func ConstSeqList(c anyvec.Creator, seqs [][]anyvec.Vector) Seq {
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
	return ConstSeq(c, batches)
}

func (c *constSeq) Creator() anyvec.Creator {
	return c.C
}

func (c *constSeq) Output() []*Batch {
	return c.Out
}

func (c *constSeq) Vars() anydiff.VarSet {
	return anydiff.VarSet{}
}

func (c *constSeq) Propagate(u []*Batch, g anydiff.Grad) {
}

// SeparateSeqs creates a separate list of vectors for
// each sequence in the batch.
func SeparateSeqs(b []*Batch) [][]anyvec.Vector {
	if len(b) == 0 {
		return nil
	}
	seqs := make([][]anyvec.Vector, len(b[0].Present))
	for _, x := range b {
		sliceSize := x.Packed.Len() / x.NumPresent()
		offset := 0
		for i, pres := range x.Present {
			if pres {
				val := x.Packed.Slice(offset, offset+sliceSize)
				offset += sliceSize
				seqs[i] = append(seqs[i], val)
			}
		}
	}
	return seqs
}
