package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestSeqSum(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		batches := []*anyseq.ResBatch{
			{
				Packed:  makeRandomVec(c, 24),
				Present: []bool{true, true, true, true},
			},
			{
				Packed:  makeRandomVec(c, 18),
				Present: []bool{true, true, true, false},
			},
			{
				Packed:  makeRandomVec(c, 12),
				Present: []bool{true, false, true, false},
			},
			{
				Packed:  makeRandomVec(c, 12),
				Present: []bool{true, false, true, false},
			},
			{
				Packed:  makeRandomVec(c, 6),
				Present: []bool{false, false, true, false},
			},
			{
				Packed:  makeRandomVec(c, 6),
				Present: []bool{false, false, true, false},
			},
		}
		var varList []*anydiff.Var
		for _, x := range batches {
			for v := range x.Packed.Vars() {
				varList = append(varList, v)
			}
		}
		inSeq := anyseq.ResSeq(batches)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anyseq.Sum(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
