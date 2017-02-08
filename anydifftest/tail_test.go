package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestTailOutput(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq := anyseq.ConstSeqList([][]anyvec.Vector{
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{0, 5})),
			},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{9, 1})),
			},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-9, 0})),
			},
		})
		actual := getComponents(anyseq.Tail(inSeq).Output())
		expected := []float64{2, -1, 0, 5, 9, 1, -9, 0}
		if !vectorsClose(actual, expected, prec) {
			t.Errorf("expected %v but got %v", expected, actual)
		}
	})
}

func TestTail(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		// Intentionally used a batch with an empty sequence.
		batches := []*anyseq.ResBatch{
			{
				Packed:  makeRandomVec(c, 18),
				Present: []bool{true, false, true, true},
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
				return anyseq.Tail(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
