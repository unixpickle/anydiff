package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestPool(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		v := makeRandomVec(c, 18)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anydiff.Pool(anydiff.Tanh(v), func(r anydiff.Res) anydiff.Res {
					return anydiff.Mul(r, r)
				})
			},
			V: []*anydiff.Var{v},
		}
		ch.FullCheck(t)
	})
}

func TestSeqPool(t *testing.T) {
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
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Pool(inSeq, func(s anyseq.Seq) anyseq.Seq {
					return anyseq.Map(s, func(v anydiff.Res, n int) anydiff.Res {
						return anydiff.Tanh(v)
					})
				})
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
