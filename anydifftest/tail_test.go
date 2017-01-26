package anydifftest

import (
	"math"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anyseq"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
)

func TestTailOutput(t *testing.T) {
	c := anyvec32.CurrentCreator()
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
	actual := anyseq.Tail(inSeq).Output().Data().([]float32)
	expected := []float32{2, -1, 0, 5, 9, 1, -9, 0}
	if len(actual) != len(expected) {
		t.Fatalf("expected length %d but got %d", len(expected), len(actual))
	}
	for i, x := range expected {
		a := actual[i]
		if math.Abs(float64(x-a)) > 1e-3 || math.IsNaN(float64(a)) {
			t.Errorf("output %d: expected %f but got %f", i, x, a)
		}
	}
}

func TestTail(t *testing.T) {
	runWithCreators(t, func(t *testing.T, c anyvec.Creator, prec float64) {
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
