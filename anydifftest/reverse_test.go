package anydifftest

import (
	"testing"

	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
)

func TestReverseOut(t *testing.T) {
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
		actual := anyseq.Reverse(inSeq)
		expected := anyseq.ConstSeqList([][]anyvec.Vector{
			{
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{0, 5})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{9, 1})),
			},
			{},
			{
				c.MakeVectorData(c.MakeNumericList([]float64{-9, 0})),
				c.MakeVectorData(c.MakeNumericList([]float64{2, -1})),
				c.MakeVectorData(c.MakeNumericList([]float64{-2, 1})),
				c.MakeVectorData(c.MakeNumericList([]float64{1, 2})),
			},
		})
		if !seqsClose(actual, expected, prec) {
			t.Errorf("expected %v but got %v", actual.Output(), expected.Output())
		}
	})
}

func TestReverseProp(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
		inSeq, varList := makeBasicTestSeqs(c)
		ch := &SeqChecker{
			F: func() anyseq.Seq {
				return anyseq.Reverse(inSeq)
			},
			V: varList,
		}
		ch.FullCheck(t)
	})
}
