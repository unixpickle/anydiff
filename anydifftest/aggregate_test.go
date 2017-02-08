package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestSeqSum(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anyseq.Sum(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}

func TestSeqSumEachOut(t *testing.T) {
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
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-9, 0})),
			},
		})
		actual := getComponents(anyseq.SumEach(inSeq).Output())
		expected := []float64{1, 2, 1, 7, 9, 1, -8, 2}
		if !vectorsClose(actual, expected, prec) {
			t.Errorf("expected %v but got %v", expected, actual)
		}
	})
}

func TestSeqSumEachProp(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &ResChecker{
			F: func() anydiff.Res {
				return anyseq.SumEach(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
