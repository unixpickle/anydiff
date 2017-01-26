package anyseq

import (
	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
)

// Tail creates a packed result with the last output from
// each sequence, ordered by sequence index (not by which
// sequence ends first).
//
// If a sequence is empty, it is ignored.
// However, at least one sequence must have at least one
// timestep.
func Tail(seq Seq) anydiff.Res {
	if len(seq.Output()) == 0 {
		panic("sequences may not all be empty")
	}
	inBatches := seq.Output()
	var outVecs []anyvec.Vector
	for i, p := range seq.Output()[0].Present {
		if !p {
			continue
		}
		t, start, end := tailVecRange(inBatches, i)
		outVecs = append(outVecs, inBatches[t].Packed.Slice(start, end))
	}
	if len(outVecs) == 0 {
		panic("sequences may not all be empty")
	}
	out := outVecs[0].Creator().Concat(outVecs...)
	return &tailRes{
		In:     seq,
		OutVec: out,
	}
}

type tailRes struct {
	In     Seq
	OutVec anyvec.Vector
}

func (t *tailRes) Output() anyvec.Vector {
	return t.OutVec
}

func (t *tailRes) Vars() anydiff.VarSet {
	return t.In.Vars()
}

func (t *tailRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	inOut := t.In.Output()
	batchUpstream := make([]*Batch, len(inOut))
	for i, x := range inOut {
		batchUpstream[i] = &Batch{
			Packed:  x.Packed.Creator().MakeVector(x.Packed.Len()),
			Present: x.Present,
		}
	}

	var upIdx int
	for i, p := range inOut[0].Present {
		if !p {
			continue
		}
		t, start, end := tailVecRange(inOut, i)

		uPart := u.Slice(upIdx, upIdx+(end-start))
		upIdx += end - start

		oldVec := batchUpstream[t].Packed
		pre := oldVec.Slice(0, start)
		post := oldVec.Slice(end, oldVec.Len())

		batchUpstream[t].Packed = uPart.Creator().Concat(pre, uPart, post)
	}

	t.In.Propagate(batchUpstream, g)
}

func tailVecRange(s []*Batch, seqIdx int) (t, startIdx, endIdx int) {
	t = len(s) - 1
	for i := 1; i < len(s); i++ {
		if !s[i].Present[seqIdx] {
			t = i - 1
			break
		}
	}

	batch := s[t]
	cols := batch.Packed.Len() / batch.NumPresent()

	for i := 0; i < seqIdx; i++ {
		if batch.Present[i] {
			startIdx += cols
		}
	}
	endIdx = startIdx + cols

	return
}
