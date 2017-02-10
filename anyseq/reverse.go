package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

type reverseRes struct {
	In  Seq
	Out []*Batch
}

// Reverse reverses each sequence in the batch to produce
// a new batch of reversed sequences.
func Reverse(s Seq) Seq {
	return &reverseRes{
		In:  s,
		Out: reverseSeqs(s.Output()),
	}
}

func (r *reverseRes) Creator() anyvec.Creator {
	return r.In.Creator()
}

func (r *reverseRes) Output() []*Batch {
	return r.Out
}

func (r *reverseRes) Vars() anydiff.VarSet {
	return r.In.Vars()
}

func (r *reverseRes) Propagate(u []*Batch, g anydiff.Grad) {
	r.In.Propagate(reverseSeqs(u), g)
}

func reverseSeqs(b []*Batch) []*Batch {
	if len(b) == 0 {
		return nil
	}
	seqs := SeparateSeqs(b)
	for _, x := range seqs {
		reverseVecs(x)
	}
	return ConstSeqList(nil, seqs).Output()
}

func reverseVecs(v []anyvec.Vector) {
	for i := 0; i < len(v)/2; i++ {
		v[i], v[len(v)-(i+1)] = v[len(v)-(i+1)], v[i]
	}
}
